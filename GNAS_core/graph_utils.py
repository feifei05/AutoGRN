import numpy as np
import pandas as pd
import torch
import dgl


MOUSE_TF_PATH = 'single_cell_type/mouse-tfs.csv'
HUMAN_TF_PATH = 'single_cell_type/human-tfs.csv'


def load_tf_gene_indices(store_path, species, gene_to_idx, gene_to_name):
    tf_path = MOUSE_TF_PATH if species == 'mouse' else HUMAN_TF_PATH
    tf_path = store_path + '/' + tf_path

    tf_df = pd.read_csv(tf_path)
    tf_names = set(tf_df['TF'].astype(str).str.upper().tolist())

    tf_indices = set()
    for tf_name in tf_names:
        if gene_to_name:
            if tf_name not in gene_to_name:
                continue
            mapped_name = gene_to_name[tf_name].upper()
        else:
            mapped_name = tf_name

        if mapped_name in gene_to_idx:
            tf_indices.add(gene_to_idx[mapped_name])

    tf_indices = sorted(tf_indices)
    print('Loaded TF list ({}) : {} TFs matched in expression matrix'.format(species, len(tf_indices)))
    return tf_indices


def build_coexpression_edges(expression_matrix, top_k=20, threshold=0.3, max_genes=None):
    expr = np.asarray(expression_matrix, dtype=np.float32)
    gene_var = expr.var(axis=1)
    valid_genes = np.where(gene_var > 1e-8)[0]

    if max_genes is not None and len(valid_genes) > max_genes:
        top_var_idx = np.argsort(gene_var[valid_genes])[-max_genes:]
        valid_genes = valid_genes[top_var_idx]

    if len(valid_genes) == 0:
        return [], []

    expr_valid = expr[valid_genes]
    expr_valid = expr_valid - expr_valid.mean(axis=1, keepdims=True)
    std = expr_valid.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    expr_valid = expr_valid / std

    corr = np.dot(expr_valid, expr_valid.T) / max(expr_valid.shape[1] - 1, 1)
    corr = np.nan_to_num(corr, nan=0.0)

    src_list = []
    dst_list = []
    local_top_k = min(top_k, corr.shape[1] - 1)

    for local_i, global_i in enumerate(valid_genes):
        scores = np.abs(corr[local_i]).copy()
        scores[local_i] = 0.0
        if local_top_k <= 0:
            continue
        neighbor_local = np.argpartition(scores, -local_top_k)[-local_top_k:]
        for local_j in neighbor_local:
            if scores[local_j] < threshold:
                continue
            global_j = valid_genes[local_j]
            src_list.append(int(global_i))
            dst_list.append(int(global_j))

    print('Co-expression edges:', len(src_list))
    return src_list, dst_list


def build_regulation_edges(gene_key_datas, gene_to_name, gene_to_idx):
    src_list = []
    dst_list = []

    for tf_block in gene_key_datas:
        for edge in tf_block:
            gene1 = edge[0].upper()
            gene2 = edge[1].upper()
            if gene_to_name:
                gene1_index = gene_to_name[gene1]
                gene2_index = gene_to_name[gene2]
            else:
                gene1_index = gene1
                gene2_index = gene2
            src_list.append(gene_to_idx[gene1_index.upper()])
            dst_list.append(gene_to_idx[gene2_index.upper()])

    print('Regulation edges:', len(src_list))
    return src_list, dst_list


def build_gene_node_features(gene_emb, tf_indices, num_genes):
    gene_feat = torch.tensor(gene_emb, dtype=torch.float32)
    is_tf = torch.zeros(num_genes, 1, dtype=torch.float32)
    if tf_indices:
        is_tf[tf_indices] = 1.0
    return torch.cat([gene_feat, is_tf], dim=1)


def build_multi_rel_gene_graph(coexpr_src, coexpr_dst, regulate_src, regulate_dst,
                               gene_feat, num_genes):
    graph_data = {}

    if len(coexpr_src) > 0:
        graph_data[('gene', 'co_expr', 'gene')] = (
            torch.tensor(coexpr_src, dtype=torch.int64),
            torch.tensor(coexpr_dst, dtype=torch.int64),
        )
    else:
        graph_data[('gene', 'co_expr', 'gene')] = (
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
        )

    if len(regulate_src) > 0:
        graph_data[('gene', 'regulates', 'gene')] = (
            torch.tensor(regulate_src, dtype=torch.int64),
            torch.tensor(regulate_dst, dtype=torch.int64),
        )
    else:
        graph_data[('gene', 'regulates', 'gene')] = (
            torch.tensor([], dtype=torch.int64),
            torch.tensor([], dtype=torch.int64),
        )

    graph = dgl.heterograph(graph_data, num_nodes_dict={'gene': num_genes})
    graph.nodes['gene'].data['feat'] = gene_feat
    graph.nodes['gene'].data['is_tf'] = gene_feat[:, -1:]
    return graph


def build_bipartite_graph(gene_have_cell, gene_feat, cell_emb):
    gene_num = gene_feat.shape[0]
    cell_num = cell_emb.shape[0]

    rating_pairs = [(i, j) for i in range(gene_num) for j in gene_have_cell[i]]
    if len(rating_pairs) == 0:
        src_gene = torch.tensor([], dtype=torch.int64)
        dst_cell = torch.tensor([], dtype=torch.int64)
    else:
        rating_pairs = torch.tensor(rating_pairs)
        src_gene = rating_pairs[:, 0]
        dst_cell = rating_pairs[:, 1]

    data_dict = {
        ('gene', 'expressed_in', 'cell'): (src_gene, dst_cell),
        ('cell', 'expressed_in', 'gene'): (dst_cell, src_gene),
    }

    bipartite_graph = dgl.heterograph(
        data_dict,
        num_nodes_dict={'gene': gene_num, 'cell': cell_num},
    )
    bipartite_graph.nodes['gene'].data['feat'] = gene_feat
    bipartite_graph.nodes['cell'].data['feat'] = torch.tensor(cell_emb, dtype=torch.float32)
    return bipartite_graph


def get_gene_features(graph):
    if isinstance(graph, dgl.DGLHeteroGraph):
        return graph.nodes['gene'].data['feat']
    return graph.ndata['feat']


def get_bipartite_features(bipartite_graph):
    return {
        'gene': bipartite_graph.nodes['gene'].data['feat'],
        'cell': bipartite_graph.nodes['cell'].data['feat'],
    }
