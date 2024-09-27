# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import argparse
from test_models import SASRecModel
from utils import EarlyStopping, get_user_seqs, check_path, set_seed, data_partition, data_partition2, data_partition3
import random
from tqdm import tqdm
from simulators_original import RecSim
import copy
def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")


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
    parser.add_argument('--env_slope', help='...', default=10, type=int)
    parser.add_argument('--k', help='...', default=10, type=int)
    parser.add_argument('--episode_length', help='...', default=20, type=int)
    parser.add_argument("--model_scheme", default="proact", type=str)
    parser.add_argument("--target_mode", default="PreMatch", type=str)

    # data augmentation args
    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="percentage of negative interactions in a sequence - robustness analysis",
    )
    parser.add_argument(
        "--training_data_ratio",
        default=1.0,
        type=float,
        help="percentage of training samples used for training - robustness analysis",
    )
    parser.add_argument(
        "--augment_type",
        default="random",
        type=str,
        help="default data augmentation types. Chosen from: \
                        mask, crop, reorder, substitute, insert, random, \
                        combinatorial_enumerate (for multi-view).",
    )
    parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
    parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
    parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator")

    ## contrastive learning task args
    parser.add_argument(
        "--temperature", default=1.0, type=float, help="softmax temperature (default:  1.0) - not studied."
    )
    parser.add_argument(
        "--n_views", default=2, type=int, metavar="N", help="Number of augmented data for each sequence - not studied."
    )
    parser.add_argument(
        "--contrast_type",
        default="Hybrid",
        type=str,
        help="Ways to contrastive of. \
                        Support InstanceCL and ShortInterestCL, IntentCL, and Hybrid types.",
    )
    parser.add_argument(
        "--num_intent_clusters",
        default="256",
        type=str,
        help="Number of cluster of intents. Activated only when using \
                        IntentCL or Hybrid types.",
    )
    parser.add_argument(
        "--seq_representation_type",
        default="mean",
        type=str,
        help="operate of item representation overtime. Support types: \
                        mean, concatenate",
    )
    parser.add_argument(
        "--seq_representation_instancecl_type",
        default="concatenate",
        type=str,
        help="operate of item representation overtime. Support types: \
                        mean, concatenate",
    )
    parser.add_argument("--warm_up_epoches", type=float, default=0, help="number of epochs to start IntentCL.")
    parser.add_argument("--de_noise", action="store_true", help="whether to de-false negative pairs during learning.")
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
    parser.add_argument("--lam", type=float, default=1.0)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--rec_weight", type=float, default=1.0, help="weight of recommendation task")
    parser.add_argument("--intent_cf_weight", type=float, default=0.1, help="weight of intent contrastive learning task")
    parser.add_argument("--cf_weight", type=float, default=0, help="weight of sequence contrastive learning task")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + ".txt"  # './data' + 'ml-1m.txt'

    # data preprocessing, to generate training data, validation, and testing data
    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)

    args.item_size = max_item + 2  ##why max_item +2  ????

    args.mask_id = max_item + 1

    # save model args
    args_str = f"{args.model_name}-{args.data_name}-{args.model_idx}"

    show_args_info(args)

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # set random seed
    random.seed(2021)

    target_file_name = f"../data/{args.data_name}/target_items.txt"

    all_target_items = []

    if args.target_mode =="random":
        random.seed(2024)
        for ii in range(args.item_size):
            all_target_items.append(ii)
        random.shuffle(all_target_items)
        target_items = all_target_items[:args.n_items]
        print(f'target items in random : {target_items}')

    elif args.target_mode =="PreMatch":
        with open(target_file_name, "r") as f:
            for line in f:
                line = line.strip()
                all_target_items.append(int(line) - 1)

        random.shuffle(all_target_items)
        if len(all_target_items)<args.n_prematch:
            pre_target_items = all_target_items
        else:
            pre_target_items = all_target_items[:args.n_prematch]
            print(f'pre matched target items: {pre_target_items}')



    model = SASRecModel(args=args).to(args.device)
    # load the best model
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()


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
    for u in range(1, args.num_users + 1):
        seq_u = user_train[u]
        # interacted[u - 1][seq_u] = 1
        if len(seq_u) > args.max_seq_length:
            seq_u = seq_u[-args.max_seq_length:]
        else:
            seq_u = seq_u + [0] * (args.max_seq_length - len(seq_u))
        seq_guidance.append(seq_u)



    if args.target_mode =="PreMatch":
        score_of_target = np.zeros(len(pre_target_items))
        for batch in range(batch_num):
            if batch == batch_num - 1:
                input_seq = seq_guidance[batch * args.batch_size:]
                interacted_input = interacted[batch * args.batch_size:]
            else:
                input_seq = seq_guidance[batch * args.batch_size: (batch + 1) * args.batch_size]
                interacted_input = interacted[batch * args.batch_size: (batch + 1) * args.batch_size]
            input_seq = copy.deepcopy(input_seq)
            input_seq = np.array(input_seq)
            pre_score = model.calc_score_for_prematch(input_seq, interacted_input,
                                                      pre_target_items).cpu().detach().numpy()
            score_of_target += np.sum(pre_score, axis=0)

        min2max_idx = np.argsort(score_of_target)
        max2min_idx = min2max_idx[::-1][:args.n_items]
        target_items = [pre_target_items[idx] for idx in max2min_idx]
        print(f'target items in pre-match manner: {target_items}')




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
        model = SASRecModel(args=args).to(args.device)
        # load the best model
        model.load_state_dict(torch.load(args.checkpoint_path))
        model.eval()
        click_logs_all = torch.zeros(args.num_users, args.episode_length)
        user_sr_all = torch.zeros(args.num_users, args.episode_length)

        env = RecSim(device='cpu', num_users=args.num_users, num_items=args.num_items, env_omega=args.env_omega, env_slope=1, env_offset=args.env_offset, boredom_decay=0.8)
        rand_array = env._dynamics_random.rand(args.n_items)

        user_emb_path = f"E:/projects/GraphAU-main/datasets/{args.data_name}/user_emb-v2-64.pt"
        item_emb_path = f"E:/projects/GraphAU-main/datasets/{args.data_name}/item_emb-v2-64.pt"

        env.reset_new()
        env.load_user_item_embeddings(user_emb_path, item_emb_path)

        ratings_logs = torch.zeros(args.num_users, args.episode_length)
        ranking_logs = torch.zeros(args.num_users, args.episode_length)
        interacted = torch.zeros(args.num_users, args.num_items + 1).to(args.device).long()


        batch_num = args.num_users // args.batch_size + 1

        cluster_cen = model.intention_cluster.weight

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

        for i in range(args.episode_length):
            recommendations = []
            coeffs = []
            for batch in range(batch_num):
                if batch == batch_num - 1:
                    input_seq = seq_guidance[batch * args.batch_size:]
                    interacted_input = interacted[batch * args.batch_size:]


                else:
                    input_seq = seq_guidance[batch * args.batch_size: (batch + 1) * args.batch_size]
                    interacted_input = interacted[batch * args.batch_size: (batch + 1) * args.batch_size]


                input_seq = copy.deepcopy(input_seq)
                input_seq = np.array(input_seq)


                ######--------------PIPG--------------
                if args.model_scheme=="PIPG":
                    recommendation, coeff = model.next_item_prediction_with_IPG3(input_seq, interacted_input,target_item, k=args.k)
                    recommendation = recommendation.cpu().numpy().tolist()
                    coeff = coeff.cpu().numpy()
                    coeff = np.around(coeff, 3)
                    coeffs.extend(coeff)

                ##---------coarse intention--------
                elif args.model_scheme=="Coarse":
                    recommendation = model.next_item_prediction_with_IPG4(input_seq, interacted_input, target_item, cluster_cen, lam=args.lam).cpu().numpy().tolist()


                ###----------coarse+PIPG
                else:
                    recommendation, coeff = model.next_item_prediction_with_ProRec(input_seq, interacted_input, target_item, cluster_cen, lam=args.lam, k=args.k)
                    recommendation = recommendation.cpu().numpy().tolist()
                    coeff = coeff.cpu().numpy()
                    coeff = np.around(coeff, 3)
                    coeffs.extend(coeff)

                recommendations.extend(recommendation)

            recommendations = torch.LongTensor(recommendations).cpu()
            recommendations[recommendations < 0] = 0

            r_target = env.get_avg_rating_new(target_item, reduce=False)

            initial_already_clicked_mask = torch.where(user_sr_all[:, 0] == 1, 0, 1)


            k = 200

            user_item_score = torch.matmul(env.user_embedd, env.item_embedd.T)
            idx_list = []
            for uu in range(args.num_users):
                temp_interacted = [x-1 for x in user_train[uu+1]]

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
            # print("in round , target item is :", i, target_item_ll[:10], ranking_logs[:10,i])

            obs = env.step_new(recommendations=recommendations.flatten(), user_mask=user_sr_all[:,i], coeffs=coeffs, noise=rand_array[count])

            for u in range(args.num_users):
                interacted[u, recommendations[u] + 1] = 1
                if obs['clicks'][u]:
                    seq_guidance[u] = [seq_guidance[u][j + 1] for j in range(len(seq_guidance[u]) - 1)] + [recommendations[u].item() + 1]
                if obs['clicks'][u] and recommendations[u].item() == target_item:
                    user_sr_all[u, i + 1:] = 1
            click_logs_all[:, i] = obs['clicks']



        for r in range(1, args.episode_length):

            temp = ratings_logs[:, r] - ratings_logs[:, 0]
            res = temp
            ratings_logs[:, r] = res



        count += 1

        all_hit_ratios.append(click_logs_all.mean().item())
        all_ratings_avg.append(ratings_logs.mean(dim=0))
        all_ranking_increase.append(ranking_logs.mean(dim=0))


    all_ratings_avg = torch.stack(all_ratings_avg)
    all_ranking_increase = torch.stack(all_ranking_increase)
    print(f'IPG, k=5, hit_ratio={torch.tensor(all_hit_ratios)[:5].mean():.4f}, IoI={(all_ratings_avg[:, 4]).mean():.4f}, IoR={(all_ranking_increase[:, 0] - all_ranking_increase[:, 4]).mean():.4f}')
    print(f'IPG, k=10, hit_ratio={torch.tensor(all_hit_ratios)[:10].mean():.4f}, IoI={(all_ratings_avg[:, 9]).mean():.4f}, IoR={(all_ranking_increase[:, 0] - all_ranking_increase[:, 9]).mean():.4f}')
    print(f'IPG, k=15, hit_ratio={torch.tensor(all_hit_ratios)[:15].mean():.4f}, IoI={(all_ratings_avg[:, 14]).mean():.4f}, IoR={(all_ranking_increase[:, 0] - all_ranking_increase[:, 14]).mean():.4f}')
    print(f'IPG, k=20, hit_ratio={torch.tensor(all_hit_ratios)[:20].mean():.4f}, IoI={(all_ratings_avg[:, 19]).mean():.4f}, IoR={(all_ranking_increase[:, 0] - all_ranking_increase[:, 19]).mean():.4f}')

if __name__=='__main__':
    main()
