# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import argparse
from test_models import SASRecModel
from utils import EarlyStopping, get_user_seqs, check_path, set_seed, data_partition, data_partition2, data_partition3, res_func
import random
from tqdm import tqdm
from simulators_original import RecSim
import copy
from mymodel.influentialRS import InfluentialNet,IRSNN
import torch.nn as nn

def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")

def load_model(config, device="cuda:0"):
    """
    Load existing nn1
    """
    # Load model information
    model_store_path = config.output_dir + config.data_name
    model_info = torch.load(model_store_path + '/irn_params64.pth.tar')

    # Restore the core module
    net = InfluentialNet(config)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)
    net.load_state_dict(model_info['state_dict'])

    # Restore the handler
    irn = IRSNN(config, net, device)
    irn.optimizer.load_state_dict(model_info['optimizer'])
    return irn

def anology_F1(p, r):
    return np.around(2*p*r / (p+r), 4)
def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="./output/", type=str)
    parser.add_argument("--data_name", default="ml-1m", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--model_idx", default=0, type=int, help="model identifier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument('--device', default='cuda:0', type=str, help='device to run the model on')
    parser.add_argument('--n_prematch', default=200, type=int, help='number of items to be set as candidate target items')
    parser.add_argument('--n_items', default=50, type=int, help='number of items to be set as target item')
    parser.add_argument('--num_users', help='...', default=945, type=int)  # 6034
    parser.add_argument('--num_items', help='...', default=2782, type=int)  # 2792,3533
    parser.add_argument('--env_omega', help='...', default=0.8, type=float)
    parser.add_argument('--env_offset', help='...', default=0.8, type=float)
    parser.add_argument('--env_slope', help='...', default=1, type=int)
    parser.add_argument('--episode_length', help='...', default=20, type=int)


    # model args
    parser.add_argument("--model_name", default="ICLRec", type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--min_seq_length", default=20, type=int)
    parser.add_argument("--lam", type=float, default=1.0)

    ####for IRN model
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--lr1', type=float, default=8e-3)  # 3e-3 for ml-1m
    parser.add_argument('--w_h', type=float, default=0.05)
    parser.add_argument('--w_obj', type=float, default=1)  # w_t always 1

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--item_emb_path", type=str, default="")


    args = parser.parse_args()



    item_emb_path = f"E:/projects/GraphAU-main/datasets/{args.data_name}/item_emb-v2-64.pt"


    args.item_emb_path = item_emb_path

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + ".txt"

    # data preprocessing, to generate training data, validation, and testing data
    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

    args.item_size = max_item + 2

    args.mask_id = max_item + 1



    # set random seed
    random.seed(2021)

    target_file_name = f"../data/{args.data_name}/target_items.txt"

    all_target_items = []


    for ii in range(args.item_size):
        all_target_items.append(ii)

    random.shuffle(all_target_items)
    target_items = all_target_items[:args.n_items]
    print(f'target items: {target_items}')

    # Load model information
    model_store_path = args.output_dir + args.data_name


    model_info = torch.load(model_store_path + '/irn_params64.pth.tar')

    net = InfluentialNet(args)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(args.device)
    net.load_state_dict(model_info['state_dict'])

    # Restore the handler
    irn = IRSNN(args, net, args.device)
    irn.optimizer.load_state_dict(model_info['optimizer'])
    model = irn    #load_model(args, device="cuda:0")



    input_file_name = f"../data/{args.data_name}/data.txt"
    if args.data_name == "steam" or args.data_name == "douban_movie":
        dataset = data_partition2(input_file_name)
    elif args.data_name == "office":
        dataset = data_partition3(input_file_name)
    else:
        dataset = data_partition(input_file_name)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    seq_guidance = []
    batch_num = args.num_users // args.batch_size + 1
    interacted = torch.zeros(args.num_users, args.num_items + 1).to(args.device).long()
    userids = torch.IntTensor([x for x in range(args.num_users)])
    for u in range(1, args.num_users + 1):
        seq_u = user_train[u]
        # interacted[u - 1][seq_u] = 1
        if len(seq_u) > args.max_seq_length:
            seq_u = seq_u[-args.max_seq_length:]
        else:
            seq_u = seq_u + [0] * (args.max_seq_length - len(seq_u))
        seq_guidance.append(seq_u)


    all_hit_ratios = []
    all_ratings_avg = []
    all_sr_avg = []
    all_ranking_increase = []
    nudge_score_avg = []

    count = 0

    for target_item in tqdm(target_items):
        input_file_name = f"../data/{args.data_name}/data.txt"
        if args.data_name =="steam" or args.data_name=="douban_movie" :
            dataset = data_partition2(input_file_name)
        elif args.data_name =="office":
            dataset = data_partition3(input_file_name)
        else:
            dataset = data_partition(input_file_name)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset

        model.eval()

        click_logs_all = torch.zeros(args.num_users, args.episode_length)
        user_sr_all = torch.zeros(args.num_users, args.episode_length)


        env = RecSim(device='cpu', num_users=args.num_users, num_items=args.num_items, env_omega=args.env_omega, env_slope=args.env_slope, env_offset=args.env_offset, boredom_decay=0.8)
        rand_array = env._dynamics_random.rand(args.n_items)

        user_emb_path = f"E:/projects/GraphAU-main/datasets/{args.data_name}/user_emb-v2-64.pt"
        item_emb_path = f"E:/projects/GraphAU-main/datasets/{args.data_name}/item_emb-v2-64.pt"
        env.reset_new()
        env.load_user_item_embeddings(user_emb_path, item_emb_path)

        ratings_logs = torch.zeros(args.num_users, args.episode_length)
        ranking_logs = torch.zeros(args.num_users, args.episode_length)
        interacted = torch.zeros(args.num_users, args.num_items + 1).to(args.device).long()


        batch_num = args.num_users // args.batch_size + 1

        seq_guidance = []

        for u in range(1, args.num_users + 1):
            seq_u = user_train[u]
            if target_item + 1 in seq_u:
                user_sr_all[u - 1] = 1
            interacted[u-1][seq_u] = 1
            if len(seq_u) > args.max_seq_length:
                seq_u = seq_u[-args.max_seq_length:]
            else:
                seq_u = seq_u + [0] * (args.max_seq_length - len(seq_u))
            seq_guidance.append(seq_u)

        sr_rate_list = []

        paths = torch.zeros((args.num_users, args.episode_length))

        for batch in range(batch_num):
            if batch == batch_num - 1:
                input_seq = seq_guidance[batch * args.batch_size:]
                interacted_input = interacted[batch * args.batch_size:]
                user_ids = user_sr_all[batch * args.batch_size:, 0]
                batch_path = paths[batch * args.batch_size:]
            else:
                input_seq = seq_guidance[batch * args.batch_size: (batch + 1) * args.batch_size]
                interacted_input = interacted[batch * args.batch_size: (batch + 1) * args.batch_size]
                user_ids = user_sr_all[batch * args.batch_size: (batch + 1) * args.batch_size, 0]
                batch_path = paths[batch * args.batch_size: (batch + 1) * args.batch_size]
            input_seq = copy.deepcopy(input_seq)

            input_seq = torch.LongTensor(np.array(input_seq)).to(args.device)
            user_ids = torch.LongTensor(np.array(user_ids)).to(args.device)

            model.get_seq_in_batch(input_seq, interacted_input, user_ids, batch_path)  # .cpu().numpy().tolist()

        for i in range(args.episode_length):
            recommendations = paths[:, i].cpu().numpy().tolist()
            recommendations = [x-1 for x in recommendations]
            recommendations = torch.LongTensor(recommendations).cpu()
            recommendations[recommendations < 0] = 0
            r_target = env.get_avg_rating_new(target_item, reduce=False)
            initial_already_clicked_mask = torch.where(user_sr_all[:, 0] == 1, 0, 1)
            k = 200

            user_item_score = torch.matmul(env.user_embedd, env.item_embedd.T)
            idx_list = []
            for uu in range(args.num_users):
                temp_interacted = [x - 1 for x in user_train[uu + 1]]
                user_item_score[uu, temp_interacted] = float('-inf')
                _, candidate_items = torch.topk(user_item_score[uu, :], k)
                idx_ = torch.nonzero(candidate_items == target_item).squeeze()
                try:
                    if idx_.shape[0] == 0:
                        idx_ = k - 1
                except IndexError:
                    idx_ = idx_.item()
                idx_list.append(idx_)
            r_ranking = np.array(idx_list)

            ratings_logs[:, i] = r_target
            ranking_logs[:, i] = torch.Tensor(r_ranking) * initial_already_clicked_mask
            coeffs = []

            obs = env.step_new(recommendations=recommendations.flatten(), user_mask=user_sr_all[:, i], coeffs=coeffs, noise=rand_array[count])

            for u in range(args.num_users):
                interacted[u, recommendations[u] + 1] = 1  # 交互过，但是可能用户不点，就会导致下次用户的点击序列中会排除掉这个物品,不合理？？
                if obs['clicks'][u]:
                    seq_guidance[u] = [seq_guidance[u][j + 1] for j in range(len(seq_guidance[u]) - 1)] + [recommendations[u].item() + 1]
                if obs['clicks'][u] and recommendations[u].item() == target_item:  # 牵引成功
                    user_sr_all[u, i + 1:] = 1

            click_logs_all[:, i] = obs['clicks']
            sr_rate = user_sr_all[:, i].mean().item()
            sr_rate_list.append(sr_rate)

        for r in range(1, args.episode_length):

            temp = (ratings_logs[:, r] - ratings_logs[:, 0]).double()
            res = temp
            ratings_logs[:, r] = res



        count += 1
        all_sr_avg.append(sr_rate_list)

        all_hit_ratios.append(click_logs_all.mean().item())
        all_ratings_avg.append(ratings_logs.mean(dim=0))

        all_ranking_increase.append(ranking_logs.mean(dim=0))

    all_ratings_avg = torch.stack(all_ratings_avg)
    all_ranking_increase = torch.stack(all_ranking_increase)


    print(
        f'IPG, k=5, hit_ratio={torch.tensor(all_hit_ratios)[:5].mean():.4f}, IoI={(all_ratings_avg[:, 4]).mean():.4f}, IoR={(all_ranking_increase[:, 0] - all_ranking_increase[:, 4]).mean():.4f}')
    print(
        f'IPG, k=10, hit_ratio={torch.tensor(all_hit_ratios)[:10].mean():.4f}, IoI={(all_ratings_avg[:, 9]).mean():.4f}, IoR={(all_ranking_increase[:, 0] - all_ranking_increase[:, 9]).mean():.4f}')
    print(
        f'IPG, k=15, hit_ratio={torch.tensor(all_hit_ratios)[:15].mean():.4f}, IoI={(all_ratings_avg[:, 14]).mean():.4f}, IoR={(all_ranking_increase[:, 0] - all_ranking_increase[:, 14]).mean():.4f}')
    print(
        f'IPG, k=20, hit_ratio={torch.tensor(all_hit_ratios)[:20].mean():.4f}, IoI={(all_ratings_avg[:, 19]).mean():.4f}, IoR={(all_ranking_increase[:, 0] - all_ranking_increase[:, 19]).mean():.4f}')

if __name__=='__main__':
    main()
