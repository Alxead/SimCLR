import argparse
import os
import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
import numpy as np
from model import Model
import torch.nn.functional as F


class MemoryBank(object):

    def __init__(self, size, dim):
        self.size = size
        self.dim = dim

    def init_bank(self, device):
        # initialize random weights
        mb_init = torch.rand(self.size, self.dim, device=device)
        std_dev = 1. / np.sqrt(self.dim / 3)
        mb_init = mb_init * (2 * std_dev) - std_dev
        mb_init = F.normalize(mb_init, dim=1)
        self.bank = mb_init.detach()

    def init_bank_from_pretrain(self, out_bank):
        self.bank = out_bank.detach()

    def update(self, indices, data_memory):
        # in lieu of scatter-update operation
        data_dim = data_memory.size(1)
        data_memory = data_memory.detach()
        indices = indices.unsqueeze(1).repeat(1, data_dim)
        self.bank = self.bank.scatter_(0, indices, data_memory)


def contrastive_loss(out_1, out_2, args):
    loss_type = args.loss
    m = args.m
    near_no = args.near_no
    loss = None

    if loss_type == 'NT-Xent':
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    elif loss_type == 'NT-Logistic':
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.mm(out, out.t().contiguous()) / temperature
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device))
        diag_r = torch.diag(torch.ones(batch_size), diagonal=batch_size).to(device)
        diag_l = torch.diag(torch.ones(batch_size), diagonal=-batch_size).to(device)
        mask = (mask - diag_r - diag_l).bool()
        # [2*B, 2*B-2]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # compute loss
        neg_loss = (-1 * torch.log(torch.sigmoid((-1 * sim_matrix)))).sum(dim=-1)
        pos_sim = torch.sum(out_1 * out_2, dim=-1) / temperature
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        pos_loss = -1 * (torch.log(torch.sigmoid(pos_sim)))
        loss = (pos_loss + neg_loss).mean()
    elif loss_type == 'Margin-Triplet':
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.mm(out, out.t().contiguous())
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device))
        diag_r = torch.diag(torch.ones(batch_size), diagonal=batch_size).to(device)
        diag_l = torch.diag(torch.ones(batch_size), diagonal=-batch_size).to(device)
        mask = (mask - diag_r - diag_l).bool()
        # [2*B, 2*B-2]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # [B,1]
        pos_sim = torch.sum(out_1 * out_2, dim=-1).unsqueeze(1)
        # [2*B, 2*B-2]
        pos_sim = pos_sim.repeat(2, 2 * batch_size - 2)
        loss_matrix = sim_matrix - pos_sim + m
        loss_matrix[loss_matrix < 0] = 0
        loss = loss_matrix.mean()
    elif loss_type == 'NT-Logistic-sh':
        # NT-Logistic semi-hard negtive mining
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.mm(out, out.t().contiguous()) / temperature
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device))
        diag_r = torch.diag(torch.ones(batch_size), diagonal=batch_size).to(device)
        diag_l = torch.diag(torch.ones(batch_size), diagonal=-batch_size).to(device)
        mask = (mask - diag_r - diag_l).bool()
        # [2*B, 2*B-2]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.sum(out_1 * out_2, dim=-1) / temperature
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        neg_mask = ((sim_matrix).min(dim=-1)[0] < pos_sim).bool()  # ensure an anchor has semi-hard negtive
        sim_matrix = sim_matrix.masked_select(neg_mask.unsqueeze(1)).view(-1, 2 * batch_size - 2)
        pos_sim = pos_sim.masked_select(neg_mask)
        sim_matrix[sim_matrix > pos_sim.unsqueeze(1)] -= int(1e12) # discard hard negtive
        hard_neg = sim_matrix.max(dim=-1)[0]
        pos_loss = -1 * (torch.log(torch.sigmoid(pos_sim)))
        neg_loss = -1 * (torch.log(torch.sigmoid(-hard_neg)))
        loss = (pos_loss + neg_loss).mean()
    elif loss_type == 'Margin-Triplet-sh':
        # Margin-Triplet semi-hard negtive mining
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.mm(out, out.t().contiguous())
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device))
        diag_r = torch.diag(torch.ones(batch_size), diagonal=batch_size).to(device)
        diag_l = torch.diag(torch.ones(batch_size), diagonal=-batch_size).to(device)
        mask = (mask - diag_r - diag_l).bool()
        # [2*B, 2*B-2]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.sum(out_1 * out_2, dim=-1)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        neg_mask = ((sim_matrix).min(dim=-1)[0] < pos_sim).bool()
        sim_matrix = sim_matrix.masked_select(neg_mask.unsqueeze(1)).view(-1, 2 * batch_size - 2)
        pos_sim = pos_sim.masked_select(neg_mask)
        sim_matrix[sim_matrix > pos_sim.unsqueeze(1)] -= int(1e12)
        hard_neg = sim_matrix.max(dim=-1)[0]
        loss_vector = hard_neg - pos_sim + m
        loss_vector[loss_vector < 0] = 0
        loss = loss_vector.mean()
    elif loss_type == 'Margin-Triplet-h':
        # Margin-Triplet hard negtive mining
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.mm(out, out.t().contiguous())
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device))
        diag_r = torch.diag(torch.ones(batch_size), diagonal=batch_size).to(device)
        diag_l = torch.diag(torch.ones(batch_size), diagonal=-batch_size).to(device)
        mask = (mask - diag_r - diag_l).bool()
        # [2*B, 2*B-2]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.sum(out_1 * out_2, dim=-1)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        hard_neg = sim_matrix.max(dim=-1)[0]
        loss_vector = hard_neg - pos_sim + m
        loss_vector[loss_vector < 0] = 0
        loss = loss_vector.mean()
    elif loss_type == 'LocalAgg':
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        back_nei_sim, back_nei_idxs = torch.topk(sim_matrix, k=near_no, dim=1)
        close_nei_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        close_nei_sim = torch.cat([close_nei_sim, close_nei_sim], dim=0)
        mask = torch.eye(2 * batch_size, device=sim_matrix.device).bool()
        self_sim = sim_matrix.masked_select(mask)
        loss = (- torch.log((close_nei_sim + self_sim) / back_nei_sim.sum(dim=-1))).mean()
    return loss


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, memory_bank, args):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    batch_i = 0
    for indices, pos_1, pos_2, target in train_bar:

        indices = indices.to(device)
        indices = torch.cat((indices, indices + traindata_len))
        pos_1, pos_2 = pos_1.to(device), pos_2.to(device)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        if args.use_bank:
            out = torch.cat([out_1, out_2], dim=0)
            all_dps = torch.exp(torch.mm(out, memory_bank.bank.t()) / temperature)
            back_nei_sim, _ = torch.topk(all_dps, k=near_no, dim=1)
            out_memory = memory_bank.bank[indices, :]
            sim_matrix = torch.exp(torch.mm(out, out_memory.t().contiguous()) / temperature)
            mask = torch.eye(2 * batch_size, device=sim_matrix.device)
            diag_r = torch.diag(torch.ones(batch_size), diagonal=batch_size).to(device)
            diag_l = torch.diag(torch.ones(batch_size), diagonal=-batch_size).to(device)
            mask = (mask + diag_r + diag_l).bool()
            close_nei_sim = sim_matrix.masked_select(mask).view(2*batch_size, -1)
            loss = (-torch.log(close_nei_sim.sum(dim=-1) / back_nei_sim.sum(dim=-1))).mean()
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()
            new_data_memory = out_memory * args.mix_coeff + (1 - args.mix_coeff) * out
            new_data_memory = F.normalize(new_data_memory)
            memory_bank.update(indices, new_data_memory)
        else:
            loss = contrastive_loss(out_1, out_2, args)
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        batch_i += 1
    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for _indexs, data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            # feature, out = net(data.cuda(non_blocking=True))
            feature, out = net(data.to(device))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for indexs_, data, _, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--loss', default='NT-Xent',
                        choices=['NT-Xent', 'NT-Logistic', 'Margin-Triplet', 'NT-Logistic-sh',
                                 'Margin-Triplet-sh', 'Margin-Triplet-h', 'LocalAgg'],
                        type=str, help='Loss function for contrastive learning')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--m', default=0.8, type=float, help='Margin for margin loss')
    parser.add_argument('--near_no', default=32, type=int, help='')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--use_bank', default=False, action='store_true')
    parser.add_argument('--mix_coeff', default=0.5, type=float)
    parser.add_argument('--pretrained', default=None, type=str)

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k, near_no, mix_coeff = args.feature_dim, args.temperature, args.k, args.near_no, args.mix_coeff
    batch_size, epochs = args.batch_size, args.epochs
    loss_type, learning_rate, margin = args.loss, args.lr, args.m
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Running on:", device)

    # data prepare
    train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    traindata_len = len(train_data)

    # model setup and optimizer config
    model = Model(feature_dim, pretrained_path=args.pretrained).to(device)
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    c = len(memory_data.classes)

    if args.use_bank:
        memory_bank = MemoryBank(size=2*traindata_len, dim=args.feature_dim)
        if args.pretrained is not None:
            all_out = []
            model.eval()
            with torch.no_grad():
                for _, data, _, target in memory_loader:
                    feature, out = model(data.to(device))
                    all_out.append(out)
                all_out = torch.cat(all_out, dim=0)
                all_out = all_out.repeat(2, 1)
            memory_bank.init_bank_from_pretrain(all_out)
        else:
            memory_bank.init_bank(device)
    else:
        memory_bank = None

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    if args.use_bank:
        save_name_pre = 'BANK_{}_lr{}_t{}_near{}_mix{}_bs{}_epoch{}'.format(
            loss_type, learning_rate, temperature, near_no, mix_coeff, batch_size, epochs)
    else:
        save_name_pre = '{}_lr{}_t{}_near{}_bs{}_epoch{}'.format(
            loss_type, learning_rate, temperature, near_no, batch_size, epochs)

    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, memory_bank, args)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
