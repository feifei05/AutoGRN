import argparse
import os
import random
import numpy as np
import torch
from data import GeneData, get_data_file_path, prepare_fold_data
from GNAS_core.model.inference_test import inference_scratch_train
from GNAS_core.device_utils import resolve_runtime_device


def config():
    parser = argparse.ArgumentParser("AutoGRN.")
    parser.add_argument("--data_name", default="mESC_1", type=str, help="The dataset name",
                        choices=['bonemarrow', 'mESC_1', 'mESC_2', 'mHSC_E', 'mHSC_GM', 'mHSC_L'])
    parser.add_argument("--device", default="0", type=int,
                        help="Running device. E.g `--device 0`, if using cpu, set `--device -1`")
    parser.add_argument("--seed", default=114514, type=int)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--epochs_scratch", type=int, default=200)

    parser.add_argument('--no_enhanced_graph', dest='use_enhanced_graph', action='store_false',
                        help='Fallback to original AutoGRN graph construction')
    parser.set_defaults(use_enhanced_graph=True)
    parser.add_argument('--coexpr_top_k', type=int, default=20)
    parser.add_argument('--coexpr_threshold', type=float, default=0.3)
    parser.add_argument('--coexpr_max_genes', type=int, default=3000)

    args = parser.parse_args()
    args.device = resolve_runtime_device(args.device)
    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_test_architecture(data_name):
    """
    手动填写各数据集的最优 GNN 架构。
    格式: [conv, bi_conv, activation, hidden_dim, fusion]
    修改对应 data_name 的列表即可复现该数据集搜索结果。
    """
    best_architectures = {
        # 来自 logger613 RL 搜索（按搜索阶段 Val AUC 最高）
        'bonemarrow': ['EdgeConv', 'BiGraphConv', 'leaky_relu', 512, 'max'],
        'mESC_1': ['GCNConv', 'BiGCNConv', 'rrelu', 256, 'abs_difference'],
        'mESC_2': ['TAGConv', 'BiGraphConv', 'leaky_relu', 256, 'sum'],  # 待搜索完成后替换
        'mHSC_E': ['TAGConv', 'BiNoneConv', 'relu', 256, 'concat'],
        'mHSC_GM': ['TAGConv', 'BiSAGEConv', 'celu', 256, 'sum'],
        'mHSC_L': ['TAGConv', 'BiNoneConv', 'relu', 256, 'concat'],
    }

    if data_name not in best_architectures:
        raise Exception('Wrong dataset name!')
    return best_architectures[data_name]


if __name__ == "__main__":
    args = config()
    set_seed(args.seed)

    args.store_path = './data_evaluation'
    args = get_data_file_path(args)
    args.tf_species = 'mouse'

    data_e = GeneData(args.rpkm_path,
                      args.label_path,
                      args.divide_path,
                      TF_num=args.TF_num,
                      gene_emb_path=args.gene_emb_path,
                      cell_emb_path=args.cell_emb_path,
                      istime=args.is_time,
                      gene_list_path=args.gene_list_path,
                      data_name=args.data_name,
                      TF_random=args.TF_random,
                      ish5=args.is_h5,
                      store_path=args.store_path,
                      use_enhanced_graph=args.use_enhanced_graph,
                      tf_species=args.tf_species,
                      coexpr_top_k=args.coexpr_top_k,
                      coexpr_threshold=args.coexpr_threshold,
                      coexpr_max_genes=args.coexpr_max_genes)

    data_e = prepare_fold_data(data_e, args)

    target_architecture = get_test_architecture(args.data_name)
    print(35 * "=" + " the testing start " + 35 * "=")
    print("dataset:", args.data_name)
    print("test gnn architecture:", target_architecture)

    val_auc, val_ap, test_auc, test_ap = inference_scratch_train(
        target_architecture,
        data_e=data_e,
        args=args,
    )
    print('Cross-Validation, Val AUC: {:.4f}, Val AP: {:.4f}'.format(val_auc, val_ap))
    print('Cross-Validation, Test AUC: {:.4f}, Test AP: {:.4f}'.format(test_auc, test_ap))
    print('=' * 70)
    print(35 * "=" + " the testing ending " + 35 * "=")
