import argparse
import os
import random
import numpy as np
import torch
from data import GeneData, get_data_file_path, prepare_fold_data
from GNAS_core.auto_model import AutoModel


def config():
    parser = argparse.ArgumentParser("AutoGRN.")
    parser.add_argument("--data_name", default="mESC_1", type=str, help="The dataset name",
                        choices=['bonemarrow', 'mESC_1', 'mESC_2', 'mHSC_E', 'mHSC_GM', 'mHSC_L'])
    parser.add_argument("--device", default="0", type=int, help="Running device. E.g `--device 0`, if using cpu, set `--device -1`")
    parser.add_argument("--seed", default=114514, type=int)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--epochs_scratch", type=int, default=200)

    parser.add_argument('--search_method', type=str, default='rl', choices=['genetic', 'rl'],
                        help='Architecture search method: genetic (original) or rl')
    parser.add_argument('--initial_num', type=int, default=100,
                        help='Random architectures for genetic search initialization')
    parser.add_argument('--search_epoch', type=int, default=6)
    parser.add_argument('--sharing_num', type=int, default=10)
    parser.add_argument('--train_epoch', type=int, default=200,
                        help='Training epochs for final scratch evaluation')
    parser.add_argument('--search_train_epoch', type=int, default=80,
                        help='Training epochs when evaluating architectures during search')
    parser.add_argument('--search_stop_num', type=int, default=5,
                        help='Early-stop patience during architecture search')
    parser.add_argument('--search_cv_folds', type=int, default=3, choices=[1, 2, 3],
                        help='CV folds used during search; final test still uses 3 folds')
    parser.add_argument('--return_top_k', type=int, default=5, help='the number of top model for testing')

    parser.add_argument('--rl_algorithm', type=str, default='ppo', choices=['ppo', 'reinforce'],
                        help='RL optimizer: ppo (Actor-Critic + clipping) or reinforce')
    parser.add_argument('--rl_warmup_num', type=int, default=30,
                        help='Random warmup architectures before RL policy learning')
    parser.add_argument('--rl_lr', type=float, default=3e-4, help='Learning rate for RL controller')
    parser.add_argument('--rl_weight_decay', type=float, default=1e-5)
    parser.add_argument('--rl_entropy_coef', type=float, default=0.01, help='Entropy bonus for exploration')
    parser.add_argument('--rl_baseline_decay', type=float, default=0.95,
                        help='EMA decay for REINFORCE baseline')
    parser.add_argument('--rl_grad_clip', type=float, default=5.0)
    parser.add_argument('--rl_ppo_clip', type=float, default=0.2, help='PPO clipping epsilon')
    parser.add_argument('--rl_ppo_epochs', type=int, default=4,
                        help='Number of PPO update passes per rollout batch')
    parser.add_argument('--rl_value_coef', type=float, default=0.5, help='Critic loss coefficient')

    parser.add_argument('--no_enhanced_graph', dest='use_enhanced_graph', action='store_false',
                        help='Fallback to original AutoGRN graph construction')
    parser.set_defaults(use_enhanced_graph=True)
    parser.add_argument('--coexpr_top_k', type=int, default=20, help='Top co-expression neighbors per gene')
    parser.add_argument('--coexpr_threshold', type=float, default=0.3, help='Co-expression correlation threshold')
    parser.add_argument('--coexpr_max_genes', type=int, default=3000,
                        help='Max variable genes used for co-expression graph construction')

    args = parser.parse_args()
    args.device = (torch.device(args.device) if args.device >= 0 else torch.device("cpu"))
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


if __name__ == "__main__":
    args = config()

    set_seed(args.seed)

    args.store_path = './data_evaluation'
    args = get_data_file_path(args)

    data_e = GeneData(args.rpkm_path,
                 args.label_path,
                 args.divide_path,
                 TF_num=args.TF_num,
                 gene_emb_path=args.gene_emb_path,
                 cell_emb_path=args.cell_emb_path,
                 istime=args.is_time, gene_list_path=args.gene_list_path,
                 data_name=args.data_name, TF_random=args.TF_random, ish5=args.is_h5,
                 store_path=args.store_path,
                 use_enhanced_graph=args.use_enhanced_graph,
                 tf_species=args.tf_species,
                 coexpr_top_k=args.coexpr_top_k,
                 coexpr_threshold=args.coexpr_threshold,
                 coexpr_max_genes=args.coexpr_max_genes)

    data_e = prepare_fold_data(data_e, args)

    args.learning_type = 'inference'
    args.data_save_name = args.data_name + '_' + args.learning_type + '_' + args.search_method

    AutoModel(data_e, args)
