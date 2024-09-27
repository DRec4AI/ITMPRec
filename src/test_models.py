# -*- coding: utf-8 -*-
#

import numpy as np
import torch
import torch.nn as nn
from modules import Encoder, LayerNorm

class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        if args.model_name=='ICLRec':
            self.intention_cluster = nn.Embedding(int(args.num_intent_clusters), args.hidden_size)


        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.device = args.device

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb





    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output


    def calc_score_for_prematch(self, log_seqs, interacted, tar):
        final_feat = self.forward(torch.LongTensor(log_seqs).to(self.device))[:, -1, :]
        # item_embs = self.item_embeddings(torch.LongTensor(range(1, self.args.item_size - 1)).to(self.device))
        item_embs = self.item_embeddings(torch.LongTensor(range(1, self.args.item_size)).to(self.device))
        logits = torch.matmul(final_feat, item_embs.T)
        logits[interacted[:, 1:] == 1] = float('-inf')

        return logits[:, tar]

    def next_item_prediction(self, log_seqs, interacted, target_item, rec2inf=False, k=50, alpha=0):
        final_feat = self.forward(torch.LongTensor(log_seqs).to(self.device))[:, -1, :]
        # item_embs = self.item_embeddings(torch.LongTensor(range(1, self.args.item_size-1)).to(self.device))
        item_embs = self.item_embeddings(torch.LongTensor(range(1, self.args.item_size)).to(self.device))
        if not rec2inf:
            # item_similarity = torch.matmul(item_embs[target_item], item_embs.T)
            logits = torch.matmul(final_feat, item_embs.T)    #+ alpha * item_similarity
            logits[interacted[:, 1:] == 1] = float('-inf')
            _, next_items = logits.max(dim=1)
        else:
            logits = torch.matmul(final_feat, item_embs.T)
            logits[interacted[:, 1:] == 1] = float('-inf')
            _, candidate_items = torch.topk(logits, k, dim=1)
            candidate_embeddings = item_embs[candidate_items]
            candidate_similarity = torch.matmul(candidate_embeddings, item_embs[target_item])
            _, next_items_index = candidate_similarity.max(dim=1)
            next_items = candidate_items[torch.arange(len(log_seqs)), next_items_index]
        return next_items  # item_id start with 0





    def target_item_for_users(self, log_seqs, target_pool_items, k=50):
        final_feat = self.forward(torch.LongTensor(log_seqs).to(self.device))[:, -1, :]
        item_embs = self.item_embeddings(torch.LongTensor(range(2, self.args.item_size)).to(self.device))

        under_target_emb = item_embs[target_pool_items]
        logits = torch.matmul(final_feat, under_target_emb.T)
        _, candidate_items = torch.topk(logits, k, largest=False, dim=1)  #修改
        target_ = np.array(target_pool_items)
        return target_[candidate_items.cpu().numpy()]  # item_id start with 0

    def next_item_prediction_with_IPG(self, log_seqs, interacted, target_item):
        final_feat = self.forward(torch.LongTensor(log_seqs).to(self.device))[:, -1, :]
        # item_embs = self.item_embeddings(torch.LongTensor(range(1, self.args.item_size-1)).to(self.device))
        item_embs = self.item_embeddings(torch.LongTensor(range(1, self.args.item_size)).to(self.device))
        # user_item_score = torch.sigmoid(torch.matmul(final_feat, item_embs.T))

        user_item_score = torch.sigmoid(torch.matmul(final_feat, item_embs.T))
        item_target_score = torch.cosine_similarity(item_embs, item_embs[target_item])
        user_target_score = torch.cosine_similarity(final_feat, item_embs[target_item]).unsqueeze(1)
        diff = item_target_score - user_target_score

        scores = user_item_score * diff

        scores[interacted[:, 1:] == 1] = float('-inf')
        _, next_items = scores.max(dim=1)
        return next_items  # item_id start with 0



    def next_item_prediction_with_IPG3(self, log_seqs, interacted, target_item, k=10):
        final_feat = self.forward(torch.LongTensor(log_seqs).to(self.device))[:, -1, :]

        item_embs = self.item_embeddings(torch.LongTensor(range(1, self.args.item_size)).to(self.device))
        user_item_score = torch.sigmoid(torch.matmul(final_feat, item_embs.T))  # [batch, n_items]

        filter_user_item_score = torch.cosine_similarity(final_feat.unsqueeze(1), item_embs.unsqueeze(0), dim=2)  #calc user item similarity

        filter_user_item_score[interacted[:, 1:] == 1] = float('-inf')
        _, hist_pref_items = filter_user_item_score.topk(k=k, dim=1)   # [batch_size, topK]


        hist_pref_item_embs = item_embs[hist_pref_items]   #[batch_size, topk, dim]
        target_item_emb = item_embs[target_item].unsqueeze(0).unsqueeze(0) #[1, 1, dim]


        temp_score = torch.cosine_similarity(hist_pref_item_embs, target_item_emb, dim=2).detach()
        hist_pre_coeff = temp_score.mean(dim=1)

        hist_pre_coeff = torch.where(hist_pre_coeff>0.2, hist_pre_coeff, 0.2).unsqueeze(1)

        item_target_score = torch.cosine_similarity(item_embs, item_embs[target_item])  # [n_items]
        user_target_score = torch.cosine_similarity(final_feat, item_embs[target_item]).unsqueeze(1)  # [batch, 1]

        guidance_score = (item_target_score - user_target_score) #* (1 - hist_pre_coff)   # diff [batch, n_items]
        scores = user_item_score * guidance_score # element wise multiplication
        scores[interacted[:, 1:] == 1] = float('-inf')
        _, next_items = scores.max(dim=1)

        return next_items, 1- hist_pre_coeff.detach().flatten()  # item_id start with 0





    def next_item_prediction_with_IPG4(self, log_seqs, interacted, target_item, cluster_cen, lam=1.0):   #with intent score

        final_feat = self.forward(torch.LongTensor(log_seqs).to(self.device))[:, -1, :]
        # item_embs = self.item_embeddings(torch.LongTensor(range(1, self.args.item_size - 1)).to(self.device))
        item_embs = self.item_embeddings(torch.LongTensor(range(1, self.args.item_size)).to(self.device))
        user_item_score = torch.sigmoid(torch.matmul(final_feat, item_embs.T))  # [batch, n_items]
        user_intention = torch.cosine_similarity(final_feat.unsqueeze(1), cluster_cen.unsqueeze(0), dim=-1)  # batch, 10
        item_centro = torch.cosine_similarity(item_embs.unsqueeze(1), cluster_cen.unsqueeze(0), dim=-1)  # batch, 10

        item_idx = torch.argmax(item_centro, dim=1)
        user_idx = torch.argmax(user_intention, dim=1)

        item_coarse_intention = cluster_cen[item_idx]      #[n_items, dim]
        user_coarse_intention = cluster_cen[user_idx]     #[batch, dim]
        new_score = torch.sigmoid(torch.matmul(user_coarse_intention, item_coarse_intention.T))

        item_target_score = torch.cosine_similarity(item_embs, item_embs[target_item])  # [n_items]
        user_target_score = torch.cosine_similarity(final_feat, item_embs[target_item]).unsqueeze(1)  # [batch, 1]
        guidance_score = (item_target_score - user_target_score)    # diff [batch, n_items]
        user_item_score += lam * new_score
        scores = user_item_score * guidance_score # element wise multiplication

        scores[interacted[:, 1:] == 1] = float('-inf')
        _, next_items = scores.max(dim=1)
        return next_items

    def next_item_prediction_with_coarse(self, log_seqs, interacted, target_item, cluster_cen, lam=1.0):  #with intent score
        final_feat = self.forward(torch.LongTensor(log_seqs).to(self.device))[:, -1, :]
        # item_embs = self.item_embeddings(torch.LongTensor(range(1, self.args.item_size - 1)).to(self.device))
        item_embs = self.item_embeddings(torch.LongTensor(range(1, self.args.item_size)).to(self.device))
        user_item_score = torch.sigmoid(torch.matmul(final_feat, item_embs.T))  # [batch, n_items]
        user_intention = torch.cosine_similarity(final_feat.unsqueeze(1), cluster_cen.unsqueeze(0), dim=-1)  # batch, 10
        item_centro = torch.cosine_similarity(item_embs.unsqueeze(1), cluster_cen.unsqueeze(0), dim=-1)  # batch, 10
        target_centro = torch.cosine_similarity(item_embs[target_item], cluster_cen.unsqueeze(0), dim=-1)  # [1, 10]
        item_idx = torch.argmax(item_centro, dim=1)
        user_idx = torch.argmax(user_intention, dim=1)
        target_idx = torch.argmax(target_centro, dim=1)


        item_coarse_intention = cluster_cen[item_idx]  # n_items, dim
        target_coarse_intention = cluster_cen[target_idx]

        user_coarse_intention = cluster_cen[user_idx]

        c_1 = torch.cosine_similarity(target_coarse_intention.unsqueeze(1), item_coarse_intention.unsqueeze(0), dim=-1)
        c_2 = torch.cosine_similarity(user_coarse_intention, target_coarse_intention, dim=-1)
        new_score = c_1 - c_2.unsqueeze(1)


        item_target_score = torch.cosine_similarity(item_embs, item_embs[target_item])  # [n_items]
        user_target_score = torch.cosine_similarity(final_feat, item_embs[target_item]).unsqueeze(1)  # [batch, 1]
        guidance_score = (item_target_score - user_target_score)  # diff [batch, n_items]


        scores = user_item_score * guidance_score  + lam * new_score# element wise multiplication

        scores[interacted[:, 1:] == 1] = float('-inf')
        _, next_items = scores.max(dim=1)
        return next_items


    def next_item_prediction_with_ProRec(self, log_seqs, interacted, target_item, cluster_cen, lam=1.0, k= 10):   #with intent score
        final_feat = self.forward(torch.LongTensor(log_seqs).to(self.device))[:, -1, :]

        item_embs = self.item_embeddings(torch.LongTensor(range(1, self.args.item_size)).to(self.device))
        user_item_score = torch.sigmoid(torch.matmul(final_feat, item_embs.T))  # [batch, n_items]


        filter_user_item_score = torch.cosine_similarity(final_feat.unsqueeze(1), item_embs.unsqueeze(0), dim=2)  #calc user item similarity

        filter_user_item_score[interacted[:, 1:] == 1] = float('-inf')
        _, hist_pref_items = filter_user_item_score.topk(k=k, dim=1)   # [batch_size, topK]


        hist_pref_item_embs = item_embs[hist_pref_items]   #[batch_size, topk, dim]
        target_item_emb = item_embs[target_item].unsqueeze(0).unsqueeze(0) #[1, 1, dim]

        temp_score = torch.cosine_similarity(hist_pref_item_embs, target_item_emb, dim=2).detach()
        hist_pre_coeff = temp_score.mean(dim=1).double()

        hist_pre_coeff = torch.where(hist_pre_coeff > 0.2, hist_pre_coeff, 0.2).unsqueeze(1)

        user_intention = torch.cosine_similarity(final_feat.unsqueeze(1), cluster_cen.unsqueeze(0), dim=-1)  # batch, 10
        item_centro = torch.cosine_similarity(item_embs.unsqueeze(1), cluster_cen.unsqueeze(0), dim=-1)  # batch, 10

        item_idx = torch.argmax(item_centro, dim=1)
        user_idx = torch.argmax(user_intention, dim=1)

        item_coarse_intention = cluster_cen[item_idx]      #[n_items, dim]
        user_coarse_intention = cluster_cen[user_idx]  #[batch, dim]
        new_score = torch.sigmoid(torch.matmul(user_coarse_intention, item_coarse_intention.T))


        item_target_score = torch.cosine_similarity(item_embs, item_embs[target_item].unsqueeze(0))  # [n_items]
        user_target_score = torch.cosine_similarity(final_feat, item_embs[target_item].unsqueeze(0)).unsqueeze(1)  # [batch, 1]
        guidance_score = (item_target_score - user_target_score)    # diff [batch, n_items]
        user_item_score += lam * new_score
        scores = user_item_score * guidance_score # element wise multiplication

        scores[interacted[:, 1:] == 1] = float('-inf')
        _, next_items = scores.max(dim=1)
        return next_items, 1- hist_pre_coeff.detach().flatten(), user_idx




    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


