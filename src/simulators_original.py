import torch
import math
from typing import List, Dict, Tuple
from tqdm import tqdm
from pathlib import Path
from collections import deque
import numpy as np
from collections import defaultdict
###from glm4flash import simple_chat  ###if you use LLM-based user click simulator, please use this line of code


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
                 sim_seed=114514, env_offset=0.7, env_slope=10, env_omega=0.8):
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

        # User preference model
        self.offset = env_offset
        self.slope = env_slope
        self.omega = env_omega


    def click_model_new(self, rels: torch.FloatTensor) -> torch.LongTensor:
        '''
            UBM click model
        '''

        clicks = torch.bernoulli(rels, generator=self.rd_gen)
        return clicks



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

    def step_new_llm(self, recommendations, user_mask, user_hist, item2names_dic, coeffs=[]):

        clicks = []
        candi_list = []
        all_users = self.num_users

        for u in range(0, all_users):

            u_can_item_name = item2names_dic[recommendations[u].item() + 1]
            candi_list.append(u_can_item_name)
            all_hist_list = user_hist[u]

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
            clicks.append(u_click)
        for u in range(0, all_users):
            if clicks[u] and user_mask[u] != 1:
                self.clicked_items[u].append(recommendations[u])
                self.all_clicked_items[u].append(recommendations[u])
                if len(coeffs) > 0:
                    self.user_embedd[u] = coeffs[u] * self.user_embedd[u] + (1 - coeffs[u]) * self.item_embedd[
                        recommendations[u]]
                else:
                    self.user_embedd[u] = self.omega * self.user_embedd[u] + (1 - self.omega) * self.item_embedd[
                        recommendations[u]]

        topic_norm = torch.linalg.norm(self.user_embedd, dim=1)
        self.user_embedd /= topic_norm.unsqueeze(1)

        obs = {'recommendations': recommendations, 'clicks': torch.IntTensor(clicks)}
        return obs

    def get_avg_rating_new(self, target_item, reduce=True):
        ratings = torch.matmul(self.user_embedd, self.item_embedd[target_item])
        return ratings.cpu()
