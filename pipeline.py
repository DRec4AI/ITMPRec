# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import os
# import wandb
import argparse

from data_provider import DataProvider, DataLoaderEvalIRS, DatasetEvalNN1, DataLoaderEvalNN1, DatasetNN, DataLoaderIRS
from src.mymodel.influentialRS import IRSNN, InfluentialNet
from utils import EarlyStopping
# from evaluator_pipeline import load_evaluator


def train_model(config, dp, device):  ###training the model
    # Load training data and validation data
    train_data, val_data = dp.get_refer_data(
                                             file_class="irs",
                                             seq_len=config.max_seq_length,
                                             min_len=config.min_seq_length,
                                             )
    # Generate dataset and dataloader
    train_dataset = DatasetNN(train_data)
    valid_dataset = DatasetNN(val_data)
    train_data_loader = DataLoaderIRS(train_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      num_workers=0)
    valid_data_loader = DataLoaderIRS(valid_dataset,
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      num_workers=0)

    # Create core model allowing for multiple GPU training
    net = InfluentialNet(config)            #影响力网络架构
    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net)
    net.to(device)

    # Create handler
    iRSNN = IRSNN(config, net, device)  #

    torch.cuda.empty_cache()

    # Early stopping handler
    model_store_path = config.output_dir + config.data_name
    if not os.path.isdir(model_store_path):
        os.makedirs(model_store_path)


    early_stopping = EarlyStopping(patience=config.early_stop_patience,verbose=True)

    # Loss function
    loss_function = nn.CrossEntropyLoss()


    example_ct = 0

    # Start training the model
    print('Start training...\n')
    print("Using device: %s" % device)
    for epoch in range(config.epochs):
        # Train model
        iRSNN.net.train()
        for i, (seqs, users) in enumerate(train_data_loader):
            seqs = seqs.to(device)
            users = users.to(device)
            loss = iRSNN.train_batch(seqs, users)

            # Update information to wandb
            example_ct += len(seqs)

            print("Epoch:", epoch, "Loss:", loss)

        # Validate model
        with torch.no_grad():
            iRSNN.net.eval()
            losses = []
            for i, (seqs, users) in enumerate(valid_data_loader):
                seqs = seqs.to(device)
                users = users.to(device)
                loss = iRSNN.get_loss_on_eval_data(seqs, users)
                losses.append(loss)

            # Calculate the average loss over the validation set
            avg_loss = sum(losses) / len(losses)
            print("Epoch: %i, Val_Loss: %f" % (epoch, avg_loss))


            # Step the learning rate scheduler
            iRSNN.pla_lr_scheduler.step(avg_loss)

            # Store the model
            model_dict = {
                'epoch': epoch,
                'state_dict': iRSNN.net.state_dict(),
                'optimizer': iRSNN.optimizer.state_dict()
            }
            if epoch > 10:  # Todo: Hyperparameter: min_epoch
                early_stopping(avg_loss, model_dict, epoch, model_store_path + '/irn_params64.pth.tar')  #modify   ####irn_params2.pth.tar, 32dimension model

                if early_stopping.early_stop:
                    print("Early stopping...")
                    print("val_loss_min:",early_stopping.val_loss_min)
                    break
    print('Finish training...\n')
def load_model(config, device="cuda:0"):
    """
    Load existing nn1
    """
    # Load model information
    model_store_path = config.output_dir + config.data_name
    model_info = torch.load(model_store_path + '/irn_params.pth.tar')

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


def test_model(config, dp, device="cuda:0"):
    print("Testing nn1...")
    print("Using device: %s" % device)
    irn = load_model(config, device)

    # Save the generated path results
    result_save_dir = "./results/" + config.dataset + '/' + config.method + '/'
    if not os.path.isdir(result_save_dir):
        os.makedirs(result_save_dir)

    eval_data = dp.get_random_evaluate_data(seq_len=100,
                                            use_train=config.use_train,
                                            file_class="irs",
                                            gap_len=config.gap_len)

    ####eval_data
    ####[new_seq(historically interacted behaviors), user_id, random_target(for guiding), label(for traditional recommendation)]

    # Random Target Task
    print("Evaluate with random target...\n")
    eval_dataset = DatasetNN(eval_data)   #[new_seq, user_id, random_target, label]
    eval_data_loader = DataLoaderEvalIRS(dataset=eval_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=False,
                                         num_workers=0,
                                         gap_len=config.gap_len,  ###gap_len=0, what is used for?
                                         seq_len=config.max_len)
    ###eval_data_loader   [new_seq, padded_seq(which comes from new_seq with head padding), user_id, random_target, label]
    n_eval = len(eval_dataset)
    with torch.no_grad():
        irn.eval()
        for i, (raw, seq, u, t, l) in enumerate(eval_data_loader):   ####raw_seqs, padded_seqs, user_ids,random_targets, labels

            # Move tensors to device
            t = t.to(device)  # Target
            l = l.to(device)  # Label
            u = u.to(device)  # User
            seq = seq.to(device)  # Sequence

            #=========!!!!!Generate the recommendation path============
            #paths, targets, actual_history, number_of_early_success_one_batch
            next_item = irn.get_seq_in_batch(seq, u, t, config.max_path_len, config.gap_len, config.sample, config.sample_k)


'''
The pipline function:
config----configuration about dataset
evaluator_config----configuration about evaluation on model
'''
def pipeline(config, evaluator_config, device='cuda:0'):
    #data processing
    dp = DataProvider(config, verbose=True)
    config.num_items = dp.n_item   #lastfm_small  2682
    config.num_users = dp.n_user  #lastfm_small  896
    # print(dp.n_user, dp.n_item)
    # exit(-9999)
    # Train nn1
    # 训练transformer的过程中不产生路径，就是学习用户交互过的一些行为序列

    train_model(config, dp, device)

    exit(-999)
    # Test nn1
    # 测试的时候既有 random_target(用于产生兴趣牵引的路径， 又有label评估模型的精度)

    test_model(config, dp, device)


