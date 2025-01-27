import copy
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, BSARecBlock

class BSARecEncoder(nn.Module):
    def __init__(self, args):
        super(BSARecEncoder, self).__init__()
        self.args = args
        block = BSARecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers

class BSARecModel(SequentialRecModel):
    def __init__(self, args):
        super(BSARecModel, self).__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = BSARecEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )               
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        return loss

    def calc_score_for_prematch(self, input_ids, interacted, tar):
        seq_output = self.forward(torch.LongTensor(input_ids).to(self.args.device))
        seq_output = seq_output[:, -1, :].to(self.args.device)
        item_emb = self.item_embeddings.weight.to(self.args.device)
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        logits[interacted[:, :] == 1] = float('-inf')

        return logits[:, tar]

    def next_item_prediction(self, log_seqs, interacted, target_item, rec2inf=False, k=50, alpha=0):
        final_feat = self.forward(torch.LongTensor(log_seqs).to(self.args.device))[:, -1, :]
        item_embs = self.item_embeddings.weight.to(self.args.device)
        if not rec2inf:
            # item_similarity = torch.matmul(item_embs[target_item], item_embs.T)
            logits = torch.matmul(final_feat, item_embs.T)  # + alpha * item_similarity
            logits[interacted[:, :] == 1] = float('-inf')
            _, next_items = logits.max(dim=1)
        else:
            logits = torch.matmul(final_feat, item_embs.T)
            logits[interacted[:, :] == 1] = float('-inf')
            _, candidate_items = torch.topk(logits, k, dim=1)
            candidate_embeddings = item_embs[candidate_items]
            candidate_similarity = torch.matmul(candidate_embeddings, item_embs[target_item])
            _, next_items_index = candidate_similarity.max(dim=1)
            next_items = candidate_items[torch.arange(len(log_seqs)), next_items_index]
        return next_items  # item_id start with 0


