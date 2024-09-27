import torch
import math
from typing import List, Dict, Tuple
from tqdm import tqdm
from pathlib import Path
from collections import deque
import numpy as np
#from my_llm_demo import simple_chat

from collections import defaultdict


def convert_dict_to_prompt(prompt_path, d):
    t = Prompt(prompt_path)
    d["historyList"] = d["historyList"].split(",") if isinstance(d["historyList"], str) else d["historyList"]
    t.historyList = d["historyList"]
    t.itemList = d["itemList"]
    return t

def process_data(dataset_name, histList, candi_item):
    # dic = {"prompt": [], "chosen": [], "rejected": []}
    # columns = list(examples.keys())
    data_point = defaultdict(list)
    data_point['historyList']= [item.strip() for item in histList]
    data_point['itemList'] = candi_item

    prompt_path = f"{dataset_name}_prompt2.txt"
    t = convert_dict_to_prompt(prompt_path, data_point)
    prompt = str(t)

    return prompt


class Prompt:
    def __init__(self, prompt_path) -> None:
        assert os.path.isfile(prompt_path), "Please specify a prompt template"
        with open(prompt_path, 'r') as f:
            raw_prompts = f.read().splitlines()
        #self.templates = [p.strip() for p in raw_prompts]
        temp_ = [p.strip() for p in raw_prompts]
        self.templates = "\n".join(temp_)

        self.historyList = []
        self.itemList = []


    def __str__(self) -> str:
        #prompt = self.templates[random.randint(0, len(self.templates) - 1)]
        prompt = self.templates

        history = "::".join(self.historyList)
        cans = "::".join(self.itemList)
        prompt = prompt.replace("[HistoryHere]", history)
        prompt = prompt.replace("[CansHere]", cans)
        prompt += " "
        return prompt


class RecSim():
    def __init__(self, topic_size=2, num_topics=10, num_items=3533, num_users=6034, device='cpu',
                 sim_seed=114514, env_offset=0.7, env_slope=10, env_omega=0.8, diversity_penalty=1.0,
                 diversity_threshold=5, recent_items_maxlen=10, short_term_boost=1., boredom_threshold=4,
                 boredom_moving_window=5, bias_penalty=0.1, boredom_decay=0.8):
        self.topic_size = topic_size
        self.num_topics = num_topics

        self.num_items = num_items
        self.num_users = num_users
        self.device = device
        self.t = 0

        self.sim_seed = sim_seed
        self.rd_gen = torch.Generator(device=device)
        self.rd_gen.manual_seed(sim_seed)
        self._dynamics_random = np.random.RandomState(sim_seed)


        self.offset = env_offset
        self.slope = env_slope
        self.omega = env_omega
        self.diversity_penalty = diversity_penalty
        self.diversity_threshold = diversity_threshold

    def click_model_new(self, rels: torch.FloatTensor) -> torch.LongTensor:
        clicks = torch.bernoulli(rels, generator=self.rd_gen)
        return clicks

    def save_env(self, path):
        # pack all the parameters of the environment
        env_params = {'user_embedd': self.user_embedd,
                      'init_user_embedd': self.init_user_embedd,
                      'item_embedd': self.item_embedd,
                      'item_comp': self.item_comp,
                      'bias': self.bias,
                      'bored': self.bored,
                      'bored_timeout': self.bored_timeout,
                      'user_short_term_comp': self.user_short_term_comp,
                      't': self.t,
                      'clicked_items': self.clicked_items,
                      'all_clicked_items': self.all_clicked_items
                      }
        torch.save(env_params, path)

    def load(self, path):
        # load all the parameters of the environment
        env_params = torch.load(path)
        self.user_embedd = env_params['user_embedd'].to(self.device)
        self.item_embedd = env_params['item_embedd'].to(self.device)
        self.init_user_embedd = env_params['init_user_embedd'].to(self.device)
        self.item_comp = env_params['item_comp'].to(self.device)
        self.bias = env_params['bias'].to(self.device)
        self.bored = env_params['bored'].to(self.device)
        self.bored_timeout = env_params['bored_timeout'].to(self.device)
        self.user_short_term_comp = env_params['user_short_term_comp'].to(self.device)
        self.t = env_params['t']
        self.clicked_items = env_params['clicked_items']
        self.all_clicked_items = env_params['all_clicked_items']

    def load_user_item_embeddings(self, user_emb_path, item_emb_path):
        if item_emb_path is not None:
            self.item_embedd = torch.nn.Embedding.from_pretrained(torch.load(item_emb_path)).weight.to(self.device)
            topic_norm = torch.linalg.norm(self.item_embedd, dim = 1)
            self.item_embedd /= topic_norm.unsqueeze(1)
        else:
            print("item embedding path is incorrect!")
            exit(-9)

        if user_emb_path is not None:
            self.user_embedd = torch.nn.Embedding.from_pretrained(torch.load(user_emb_path)).weight.to(self.device)
            topic_norm = torch.linalg.norm(self.user_embedd, dim = 1)
            self.user_embedd /= topic_norm.unsqueeze(1)
        else:
            print("user embedding path is incorrect!")
            exit(-9)


    def reset_new(self):
        self.clicked_items = [deque([], self.recent_items_maxlen) for _ in range(self.num_users)]
        self.all_clicked_items = [[] for _ in range(self.num_users)]

    def step_new(self, recommendations, user_mask, coeffs, noise):
        ## Compute relevances
        item_embedings = self.item_embedd[recommendations]
        score = torch.sum(item_embedings * self.user_embedd, dim=1)
        relevances = 1 / (1 + torch.exp(-(score - self.offset) * self.slope))    ## Rescale relevance
        if noise is not None:
            relevances += noise/100

        relevances = relevances.double()
        relevances = torch.where(relevances>1.0, 1.0, relevances)
        ## Interaction
        clicks = self.click_model_new(relevances)

        for u in range(self.num_users):
            if clicks[u] and user_mask[u]!=1:
                self.clicked_items[u].append(recommendations[u])
                self.all_clicked_items[u].append(recommendations[u])
                if len(coeffs)>0:
                    self.user_embedd[u] = coeffs[u] * self.user_embedd[u] + (1 - coeffs[u]) * self.item_embedd[recommendations[u]]
                else:
                    self.user_embedd[u] = self.omega * self.user_embedd[u] + (1 - self.omega) * self.item_embedd[recommendations[u]]

        topic_norm = torch.linalg.norm(self.user_embedd, dim=1)
        self.user_embedd /= topic_norm.unsqueeze(1)

        obs = {'recommendations': recommendations, 'clicks': clicks}
        return obs




    def step_llm(self, recommendations, user_hist, interacted_item, item2names_dic, all_users, coeffs=[]):
        self.t += 1
        info = {}
        self.bored_timeout -= self.bored.long()
        self.bored = self.bored & (self.bored_timeout != 0)
        self.bored_timeout[self.bored == False] = 5
        canid_list = []
        # all_clicked_items_lists = []

        all_users = 20
        clicks = []
        for u in range(0, all_users):
            u_can_item_name = item2names_dic[recommendations[u].item() + 1]
            canid_list.append(u_can_item_name)
            all_hist_list = user_hist[u]

            user_clicked_name_list = []

            interacted_item_list_u = interacted_item[u, :]
            # specific_value = 1
            # indices = torch.nonzero(torch.eq(interacted_item_list_u, specific_value)).squeeze(1).tolist()
            for item in all_hist_list[-20:]:
                if item != 0:
                    user_clicked_name_list.append(item2names_dic[item])

            u_prompt = process_data(self.dataset_name, user_clicked_name_list, [u_can_item_name])
            u_click = simple_chat(u_prompt, use_stream=False)

            if u % 200 == 0:
                print("user click {} feedback is {}".format(u, u_click))
            if "A: 1" in u_click:
                u_click = 1
            else:
                u_click = 0

            clicks.append(u_click)

        for u in range(self.num_users):
            if clicks[u] and user_mask[u]!=1:
                self.clicked_items[u].append(recommendations[u])
                self.all_clicked_items[u].append(recommendations[u])
                if len(coeffs)>0:
                    self.user_embedd[u] = coeffs[u] * self.user_embedd[u] + (1 - coeffs[u]) * self.item_embedd[recommendations[u]]
                else:
                    self.user_embedd[u] = self.omega * self.user_embedd[u] + (1 - self.omega) * self.item_embedd[recommendations[u]]
                # if self.bored[u, self.user_short_term_comp[u]] > 0:
                #     self.user_short_term_comp[u] = self.item_comp[recommendations[u]]
                # self.bias[u, recommendations[u]] += self.bias_penalty

        topic_norm = torch.linalg.norm(self.user_embedd, dim=1)
        self.user_embedd /= topic_norm.unsqueeze(1)
        obs = {'recommendations': recommendations, 'clicks': clicks}
        return obs


    def step_new_one(self, recommendations, u, user_mask_u, coeffs, noise):
        ## Compute relevances
        item_embedings = self.item_embedd[recommendations]
        score = torch.sum(item_embedings * self.user_embedd, dim=1)
        relevances = 1 / (1 + torch.exp(-(score - self.offset) * self.slope))    ## Rescale relevance
        if noise is not None:
            relevances += noise/100

        relevances = relevances.double()
        relevances = torch.where(relevances>1.0, 1.0, relevances)
        ## Interaction
        clicks = self.click_model_new(relevances)
        if clicks[u] and user_mask_u!=1:
            self.clicked_items[u].append(recommendations[u])
            self.all_clicked_items[u].append(recommendations[u])
            if len(coeffs)>0:
                self.user_embedd[u] = coeffs[u] * self.user_embedd[u] + (1 - coeffs[u]) * self.item_embedd[recommendations[u]]
            else:
                self.user_embedd[u] = self.omega * self.user_embedd[u] + (1 - self.omega) * self.item_embedd[recommendations[u]]

        topic_norm = torch.linalg.norm(self.user_embedd, dim=1)
        self.user_embedd /= topic_norm.unsqueeze(1)

        obs = {'recommendations': recommendations[u], 'clicks': clicks[u]}
        return obs

    def step_new_llm_one(self, recommendations_u, u, user_mask_u, user_hist_u, item2names_dic, coeffs=[]):


        clicks_u = []
        candi_list = []


        u_can_item_name = item2names_dic[recommendations_u.item() + 1]
        candi_list.append(u_can_item_name)
        all_hist_list = user_hist_u

        user_clicked_name_list = []


        hist_count = 0
        hist_len = len(all_hist_list) - 1
        for kk in range(hist_len, -1, -1):
            if all_hist_list[kk] != 0:
                hist_count += 1
                user_clicked_name_list.append(item2names_dic[all_hist_list[kk]])
                if hist_count > 20:
                    break

        u_prompt = process_data(self.data_name, user_clicked_name_list, [u_can_item_name])
        u_click = simple_chat(u_prompt, use_stream=False)


        if "A: 1" in u_click:
            u_click = 1
        else:
            u_click = 0

        if u_click and user_mask_u != 1:
            self.clicked_items[u].append(recommendations_u)
            self.all_clicked_items[u].append(recommendations_u)
            if len(coeffs) > 0:
                self.user_embedd[u] = coeffs[u] * self.user_embedd[u] + (1 - coeffs[u]) * self.item_embedd[recommendations_u]
            else:
                self.user_embedd[u] = self.omega * self.user_embedd[u] + (1 - self.omega) * self.item_embedd[recommendations_u]

        topic_norm = torch.linalg.norm(self.user_embedd, dim=1)
        self.user_embedd /= topic_norm.unsqueeze(1)

        obs = {'recommendations': recommendations_u, 'clicks': torch.IntTensor(u_click)}
        return obs





    def get_avg_rating(self, target_item, reduce=True):
        bias = self.bias[:, target_item].flatten()
        ratings = torch.matmul(self.user_embedd, self.item_embedd[target_item])
        ratings -= bias
        if reduce:
            return torch.mean(ratings).item()
        else:
            return ratings.cpu()

    def get_increase_of_ranking(self, target_item, interacted, k=200):
        user_item_score = torch.matmul(self.user_embedd, self.item_embedd.T)
        idx_list = []
        for u in range(self.num_users):
            user_item_score[u, interacted[u, :] == 1] = float('-inf')
            _, candidate_items = torch.topk(user_item_score[u, :], k)
            idx_ = torch.nonzero(candidate_items == target_item).squeeze()
            try:
                if idx_.shape[0] == 0:
                    idx_ = k - 1
            except IndexError:
                idx_ = idx_.item()
            idx_list.append(idx_)

        return np.array(idx_list)

    def get_avg_rating_new(self, target_item, reduce=True):
        ratings = torch.matmul(self.user_embedd, self.item_embedd[target_item])
        return ratings.cpu()


    def get_avg_rating_new_batch(self, target_items, reduce=True):
        # print('shape of bias:', self.bias.shape)   #(n_users, n_items)
        # all_user_idx = np.arange(0, len(target_items))  #(n_users, )
        ####target_items.shape   [n_users, ]
        ratings = torch.sum(self.user_embedd * self.item_embedd[target_items], dim=1)  # [n_users,20], [20]
        if reduce:
            return torch.mean(ratings).item()
        else:
            return ratings.cpu()

    def get_all_scores(self):
        scores = torch.matmul(self.user_embedd, self.item_embedd.T) - self.bias
        return scores