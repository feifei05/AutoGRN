import torch
import torch.nn as nn
from GNAS_core.search_space.search_space_utils import conv_map, hetero_gene_conv_map, bi_conv_map, act_map, fusion_map
from GNAS_core.graph_utils import get_gene_features, get_bipartite_features


class GNN_Model(nn.Module):
    def __init__(self, sample_architecture, in_dim, out_dim=1, layer=3, use_hetero_gene_graph=True,
                 gene_cell_etype='expressed_in', cell_in_dim=None, bipartite_gene_in_dim=None):
        super(GNN_Model, self).__init__()
        self.sample_architecture = sample_architecture
        self.in_dim = in_dim
        self.cell_in_dim = cell_in_dim if cell_in_dim is not None else in_dim
        self.bipartite_gene_in_dim = bipartite_gene_in_dim if bipartite_gene_in_dim is not None else self.cell_in_dim
        self.out_dim = out_dim
        self.layer = layer
        self.use_hetero_gene_graph = use_hetero_gene_graph
        self.gene_cell_etype = gene_cell_etype

        self.conv_type = self.sample_architecture[0]
        self.bi_conv_type = self.sample_architecture[1]
        self.act_type = self.sample_architecture[2]
        self.hidden_dim = int(self.sample_architecture[3])
        self.fusion_type = self.sample_architecture[4]

        self.cells = []
        gene_input_dim = self.in_dim
        bipartite_gene_input_dim = self.bipartite_gene_in_dim
        cell_input_dim = self.cell_in_dim
        for i in range(self.layer):
            cell = GNN_Cell(
                self.conv_type,
                self.bi_conv_type,
                self.act_type,
                self.hidden_dim,
                gene_input_dim,
                bipartite_gene_in_dim=bipartite_gene_input_dim,
                cell_input_dim=cell_input_dim,
                use_hetero_gene_graph=use_hetero_gene_graph,
                gene_cell_etype=gene_cell_etype,
            )
            self.add_module("cell{}".format(i), cell)
            self.cells.append(cell)
            gene_input_dim = self.hidden_dim
            bipartite_gene_input_dim = self.hidden_dim
            cell_input_dim = self.hidden_dim

        self.post_processing = Post_Pro(self.fusion_type, self.hidden_dim, self.out_dim, self.layer)

    def forward(self, graph, bipartite_graph, samples):
        reprs = []
        node_emb = get_gene_features(graph)
        bipartite_node_emb = get_bipartite_features(bipartite_graph)

        for i in range(self.layer):
            node_emb, bipartite_node_emb = self.cells[i](graph, node_emb, bipartite_graph, bipartite_node_emb)
            reprs.append(node_emb)

        node_emb = torch.cat(reprs, dim=1)
        score = self.post_processing(node_emb, samples)
        return score


class GNN_Cell(nn.Module):
    def __init__(self, conv_type, bi_conv_type, act_type, hidden_dim, in_dim,
                 bipartite_gene_in_dim=None, cell_input_dim=None,
                 use_hetero_gene_graph=True, gene_cell_etype='expressed_in'):
        super().__init__()
        self.conv_type = conv_type
        self.bi_conv_type = bi_conv_type
        self.act_type = act_type
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.bipartite_gene_in_dim = bipartite_gene_in_dim if bipartite_gene_in_dim is not None else in_dim
        self.cell_in_dim = cell_input_dim if cell_input_dim is not None else self.bipartite_gene_in_dim
        self.use_hetero_gene_graph = use_hetero_gene_graph

        has_bi_conv = self.bi_conv_type != 'BiNoneConv'
        gene_out_dim = self.hidden_dim // 2 if has_bi_conv else self.hidden_dim

        if self.use_hetero_gene_graph:
            self.conv = hetero_gene_conv_map(self.conv_type, self.in_dim, gene_out_dim)
        else:
            self.conv = conv_map(self.conv_type, self.in_dim, gene_out_dim)

        self.bipartite_conv = None
        if has_bi_conv:
            self.bipartite_conv = bi_conv_map(
                self.bi_conv_type,
                self.bipartite_gene_in_dim,
                self.cell_in_dim,
                self.hidden_dim // 2,
                gene_cell_etype=gene_cell_etype,
            )

        self.act = act_map(self.act_type)

    def _apply_gene_conv(self, graph, node_emb):
        if self.use_hetero_gene_graph:
            node_dict = self.conv(graph, {'gene': node_emb})
            node_emb = node_dict['gene']
        else:
            node_emb = self.conv(graph, node_emb)
        if len(node_emb.shape) == 3:
            node_emb = node_emb.mean(dim=1)
        return self.act(node_emb)

    def forward(self, graph, node_emb, bipartite_graph, bipartite_node_emb):
        node_emb = self._apply_gene_conv(graph, node_emb)

        if self.bipartite_conv is not None:
            bipartite_node_emb = self.bipartite_conv(bipartite_graph, bipartite_node_emb)
            bipartite_node_emb = {
                key: value.mean(dim=1) if len(value.shape) == 3 else value
                for key, value in bipartite_node_emb.items()
            }
            node_emb = torch.cat([node_emb, bipartite_node_emb['gene']], dim=1)
            bipartite_node_emb['gene'] = node_emb

        return node_emb, bipartite_node_emb


class Post_Pro(nn.Module):
    def __init__(self, fusion_type, hidden_dim, out_dim, layer):
        super().__init__()
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layer = layer
        self.src_process = nn.Sequential(
            nn.Linear(self.hidden_dim * self.layer, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.dst_process = nn.Sequential(
            nn.Linear(self.hidden_dim * self.layer, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.fusion = fusion_map(self.fusion_type, self.hidden_dim, self.out_dim)

    def forward(self, node_emb, samples):
        src, dst = samples[:, 0], samples[:, 1]
        src_emb = self.src_process(node_emb[src])
        dst_emb = self.dst_process(node_emb[dst])
        score = self.fusion(src_emb, dst_emb)
        return score
