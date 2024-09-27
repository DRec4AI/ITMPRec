import os

import torch
# import wandb

os.environ["WANDB_API_KEY"] = "053079ba1b504a9e8eaaa5704e1319753430c125"
# os.environ["WANDB_MODE"] = 'offline'
# wandb.login()

from pipeline import pipeline
from model_params import irs_sweep_config, NetParams
from utils import str2bool, set_seed
import argparse

parser = argparse.ArgumentParser()
# system args
parser.add_argument("--data_dir", default="../data/", type=str)
parser.add_argument("--output_dir", default="./src/output/", type=str)  # original is 'output'
parser.add_argument("--data_name", default="ml-1m", type=str)  # original 'Sports_and_Outdoors'
parser.add_argument("--do_eval", action="store_true")
parser.add_argument("--model_idx", default=5, type=int, help="model idenfier 10, 20, 30...")
parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
parser.add_argument('--device', default='cuda:0', type=str, help='device to run the model on')
parser.add_argument('--num_users', help='...', default=945, type=int)  # 6034
parser.add_argument('--num_items', help='...', default=2782, type=int)  # 2792,3533
parser.add_argument('--early_stop_patience', type=int, default=10)

# data augmentation args
parser.add_argument(
    "--noise_ratio",
    default=0.0,
    type=float,
    help="percentage of negative interactions in a sequence - robustness analysis",
)


parser.add_argument("--hidden_size", type=int, default=64,help="hidden size of transformer model")  # modify, original 64
####for IRN model
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--ffn_dim', type=int, default=256)
parser.add_argument('--lr1', type=float, default=8e-3)  # 3e-3 for ml-1m
parser.add_argument('--w_h', type=float, default=0.05)
parser.add_argument('--w_obj', type=float, default=1)  # w_t always 1

parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
parser.add_argument("--num_attention_heads", default=2, type=int)
parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
parser.add_argument("--initializer_range", type=float, default=0.02)
parser.add_argument("--max_seq_length", default=50, type=int)
parser.add_argument("--min_seq_length", default=20, type=int)
parser.add_argument("--lam", type=float, default=1.0)

# train args
parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--item_emb_path",type=str, default="")
args = parser.parse_args()


# Evaluator Configuration
# Here we use SampleNet as an example
params = NetParams()
tran_config = getattr(params, "params_tran")  # parameters of model

# Device control
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
# Ensure deterministic behavior
set_seed(args.seed, cuda=torch.cuda.is_available())


if __name__ == '__main__':

    item_emb_path = f"E:/projects/GraphAU-main/datasets/{args.data_name}/item_emb-v2-64.pt"
    args.item_emb_path = item_emb_path
    pipeline(args, tran_config, device=device)





