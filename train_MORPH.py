import numpy as np
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from dataset import MORPHDataset
import matplotlib.pyplot as plt
import pandas as pd
import random
from model import ResNet101
from model import *
from vig import *
from torch_geometric.utils import degree
from gcn_lib.torch_edge import DenseDilatedKnnGraph
from thop import profile


import warnings
warnings.filterwarnings("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.2):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MORPH_loaders(batch_size=256, valid=0.2, num_workers=0, pin_memory=False):
    trainset = MORPHDataset()

    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory, drop_last=True)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory, drop_last=True)

    return train_loader, valid_loader


def CS_Score(pred, label, L):
    num = len(pred)
    dif = abs(pred - label)
    score = 0
    for i in dif:
        if i <= L:
            score += 1
    score = score / num
    return score

def train_or_eval_graph_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    mse_losses = []
    total_losses = []
    preds = []
    labels = []

    my_margin = args.margin1
    my_margin_2 = my_margin + args.margin2
    margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
    num_neg = args.NN
    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()
    
    cs = 0
    lbl_z = torch.tensor([0.]).cuda()
    for image, label in dataloader:
        if train:
            optimizer.zero_grad()

        if cuda:
            image, label = image.cuda(), label.cuda()

        idx_list = []
        h_a, h_p, edge_index, pred = model(image)  # batch_size * 1
        _, _, nb_nodes, _ = edge_index.shape
        edge_index = edge_index.view(2, -1)
        A_degree = degree(edge_index[1], nb_nodes, dtype=int).tolist()
        for i in range(num_neg):
            idx_0 = np.random.permutation(nb_nodes)
            idx_list.append(idx_0)
        deg_list_2 = []
        deg_list_2.append(0)
        for i in range(nb_nodes):
            deg_list_2.append(deg_list_2[-1] + A_degree[i])
        idx_p_list = []
        edge_index = edge_index[0]
        for j in range(1, 102):
            random_list = [deg_list_2[i] + j % A_degree[i] for i in range(nb_nodes)]
            idx_p = edge_index[random_list]
            idx_p_list.append(idx_p)
        idx_list = []
        for i in range(num_neg):
            idx_0 = np.random.permutation(nb_nodes)
            idx_list.append(idx_0)

        h_p_1 = (h_a[idx_p_list[epoch % 100]] + h_a[idx_p_list[(epoch + 2) % 100]] + h_a[
            idx_p_list[(epoch + 4) % 100]] + h_a[idx_p_list[(epoch + 6) % 100]] + h_a[
                     idx_p_list[(epoch + 8) % 60]]) / 5
        s_p = F.pairwise_distance(h_a, h_p)
        s_p_1 = F.pairwise_distance(h_a, h_p_1)
        s_n_list = []

        for h_n in idx_list:
            s_n = F.pairwise_distance(h_a, h_a[h_n])
            s_n_list.append(s_n)
        margin_label = -1 * torch.ones_like(s_p)

        loss_mar = 0
        loss_mar_1 = 0
        mask_margin_N = 0
        for s_n in s_n_list:
            loss_mar += (margin_loss(s_p, s_n, margin_label)).mean()
            loss_mar_1 += (margin_loss(s_p_1, s_n, margin_label)).mean()
            mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
        mask_margin_N = mask_margin_N / num_neg

        output = pred.view(-1)
        label = label.to(torch.float32)
        cs += CS_Score(output, label, 1)
        mse_loss = loss_function(output, label)
        # print("output", output[:10])
        # print("label", label[:10])
        # print("mse", mse_loss)
        loss = loss_mar * args.w_loss1 + loss_mar_1 * args.w_loss2 + mask_margin_N * args.w_loss3 + mse_loss

        mse_losses.append(mse_loss.cpu().item())
        total_losses.append(loss.cpu().item())
        labels.append(label.data.cpu().numpy())
        preds.append(output.data.cpu().numpy())

        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()


    avg_loss = np.sum(mse_losses) / len(dataloader)
    cs = cs / len(dataloader) 

    return avg_loss, preds, labels, cs


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    mse_losses = []
    total_losses = []
    preds = []
    labels = []
    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    cs = 0
    for image, label in dataloader:
        if train:
            optimizer.zero_grad()

        if cuda:
            image, label = image.cuda(), label.cuda()

        pred = model(image)

        output = pred.view(-1)
        label = label.to(torch.float32)
        cs += CS_Score(output, label, 1)
        mse_loss = loss_function(output, label)

        mse_losses.append(mse_loss.cpu().item())
        labels.append(label.data.cpu().numpy())
        preds.append(output.data.cpu().numpy())

        if train:
            mse_loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    avg_loss = np.sum(mse_losses) / len(dataloader)
    cs = cs / len(dataloader)

    return avg_loss, preds, labels, cs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.6, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=196, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weight')
    parser.add_argument('--attention', action='store_true', default=False, help='use attention on top of lstm')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--w_loss1', type=float, default=1, help='')
    parser.add_argument('--w_loss2', type=float, default=1, help='')
    parser.add_argument('--w_loss3', type=float, default=1, help='')
    parser.add_argument('--margin1', type=float, default=0.8, help='')
    parser.add_argument('--margin2', type=float, default=0.2, help='')
    parser.add_argument('--NN', type=int, default=4, help='Negative sample')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')


    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    batch_size = args.batch_size
    cuda = args.cuda
    n_epochs = args.epochs

    model = SUGRL_Fast(cfg=[512, 512])
    if cuda:
        model.cuda()

    input = torch.randn(1, 3, 224, 224)
    input = input.cuda()
    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)


    loss_function = nn.L1Loss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)

    train_loader, valid_loader = get_MORPH_loaders(batch_size=args.batch_size, valid=0.2, num_workers=0, pin_memory=False)

    best_loss, best_labels, best_preds = None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_mse_loss, train_preds, train_labels, train_cs = train_or_eval_model(model, loss_function,
                                                                              train_loader, e, optimizer, True)
        valid_mse_loss, valid_preds, val_labels, valid_cs = train_or_eval_model(model, loss_function, valid_loader, e)

        if best_loss == None or best_loss > valid_mse_loss:
            best_loss, best_labels, best_preds = valid_mse_loss, val_labels, valid_preds

        if args.tensorboard:
            writer.add_scalar('valid: mse_loss', train_mse_loss, e)
            writer.add_scalar('train: mse_loss', valid_mse_loss, e)
        # print(
        #     'epoch {} train_mse_loss {} train_preds {} train_labels {} valid_mse_loss {} valid_preds {} val_labels {} time {}'. \
        #     format(e + 1, train_mse_loss, train_preds, train_labels, valid_mse_loss, valid_preds, val_labels, \
        #           round(time.time() - start_time, 2)))
        print(
            'epoch {} train_mae_loss {}  train_cs{} valid_mae_loss {} valid_cs {} time {}'.format(e + 1, train_mse_loss, train_cs, valid_mse_loss, valid_cs, round(time.time() - start_time, 2)))

    print("best_mae", best_loss)
    if args.tensorboard:
        writer.close()
