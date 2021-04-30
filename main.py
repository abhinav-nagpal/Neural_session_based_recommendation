import torch
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import csv
import operator
import os
import time
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join
from torch.utils.data import Dataset
def get_data(base_path, val_split=0.1, maxi=19):
    
    train_path = base_path + 'train.txt'
    test_path = base_path + 'test.txt'
    with open(train_path, 'rb') as f1:
        train = pickle.load(f1)

    with open(test_path, 'rb') as f2:
        test = pickle.load(f2)

    if maxi:
        train_x = []
        train_y = []
        for i, j in zip(train[0], train[1]):
            if len(i) < maxi:
                train_x.append(i)
                train_y.append(j)
            else:
                train_x.append(i[:maxi])
                train_y.append(j)
        train = (train_x, train_y)

        test_x = []
        test_y = []
        for i, j in zip(test[0], test[1]):
            if len(i) < maxi:
                test_x.append(i)
                test_y.append(j)
            else:
                test_x.append(i[:maxi])
                test_y.append(j)
        test = (test_x, test_y)

    k = len(train_x)
    arr = np.arange(k, dtype='int32')
    np.random.shuffle(arr)
    train_count = int(np.round(k * (1.0 - val_split)))
    val_x = [train_x[i] for i in arr[train_count:]]
    val_y = [train_y[i] for i in arr[train_count:]]
    train_x = [train_x[i] for i in arr[:train_count]]
    train_y = [train_y[i] for i in arr[:train_count]]

    train = (train_x, train_y)
    val = (val_x, val_y)
    test = (test_x, test_y)

    return train, val, test

class Recsys(Dataset):

    def __init__(self, x):
        self.x = x
        
    def __getitem__(self, index):
        sess = self.x[0][index]
        targ = self.x[1][index]
        return sess, targ

    def __len__(self):
        return len(self.x[0])

def combine(x):

    x.sort(key=lambda i: len(i[0]), reverse=True)
    lens = [len(s) for s, _ in x]
    y = []
    session = torch.zeros(len(x), max(lens)).long()
    for i, (s, l) in enumerate(x):
        session[i,:lens[i]] = torch.LongTensor(s)
        y.append(l)
    
    session = session.transpose(0,1)
    return session, torch.tensor(y).long(), lens

class MODEL(nn.Module):

    def __init__(self, n_items, hidden, emb_dim, batch):

        super(MODEL, self).__init__()
        self.n_items = n_items
        self.hidden = hidden
        self.batch = bs
        self.n_layers = 1
        self.emb_dim = emb_dim
        self.emb = nn.Embedding(self.n_items, self.emb_dim, padding_idx = 0)
        self.emb_dt = nn.Dropout(0.25)
        self.gru = nn.GRU(self.emb_dim, self.hidden, self.n_layers)
        self.l1 = nn.Linear(self.hidden, self.hidden, bias=False)
        self.l2 = nn.Linear(self.hidden, self.hidden, bias=False)
        self.l3 = nn.Linear(self.hidden, 1, bias=False)
        self.dt = nn.Dropout(0.5)
        self.l5 = nn.Linear(self.emb_dim, 2 * self.hidden, bias=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, seq, lengths):
        intermediate = self.init_hidden(seq.size(1))
        emb_layer = self.emb_dt(self.emb(seq))
        emb_layer = pack_padded_sequence(emb_layer, lengths)
        gru_out, intermediate = self.gru(emb_layer, intermediate)
        gru_out, lengths = pad_packed_sequence(gru_out)

        last = intermediate[-1]
        gru_out = gru_out.permute(1, 0, 2)

        k1 = last
        layer1 = self.l1(gru_out.contiguous().view(-1, self.hidden)).view(gru_out.size())  
        layer2 = self.l2(last)

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device = self.device), torch.tensor([0.], device = self.device))
        layer2_modified = layer2.unsqueeze(1).expand_as(layer1)
        layer2_modified2 = mask.unsqueeze(2).expand_as(layer1) * layer2_modified

        alpha = self.l3(torch.sigmoid(layer1 + layer2_modified2).view(-1, self.hidden)).view(mask.size())
        k2 = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        k = torch.cat([k2, k1], 1)
        k = self.dt(k)
        
        item_embs = self.emb(torch.arange(self.n_items).to(self.device))
        scores = torch.matmul(k, self.l5(item_embs).permute(1, 0))

        return scores

    def init_hidden(self, batch):
        return torch.zeros((self.n_layers, batch, self.hidden), requires_grad=True).to(self.device)

def calculate_recall(ind, y):

    y = y.view(-1, 1)
    y = y.expand_as(ind)

    match = (y == ind)
    match = match.nonzero()

    if len(match) == 0:
        return 0

    num_match = (y == ind)
    num_match = num_match.nonzero()[:, :-1].size(0)

    return float(num_match) / y.size(0)

def calculate_mrr(ind, y):

    y = y.view(-1, 1)
    y = y.expand_as(ind)
    
    match = (y == ind)
    match = match.nonzero()

    ranks = (match[:, -1] + 1).float()
    ranks = torch.reciprocal(ranks)

    return (torch.sum(ranks).data / y.size(0)).item()

def get_metrics(ind, y, k=20):
    _, ind = torch.topk(ind, k, -1)
    recall = calculate_recall(ind, y)
    mrr = calculate_mrr(ind, y)
    return recall, mrr

def epochTrain(dataloader_train, model, opt, epoch, num_epochs, loss_fn, log_aggr=1):
    model.train()

    total_loss = 0

    start = time.time()
    for i, (seq, y, l) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
        seq = seq.to(device)
        y = y.to(device)
        
        opt.zero_grad()
        y_preds = model(seq, l)
        loss = loss_fn(y_preds, y)
        loss.backward()
        opt.step() 

        loss = loss.item()
        total_loss += loss

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss, total_loss / (i + 1),
                  len(seq) / (time.time() - start)))

        start = time.time()

def validate(dataloader_val, model):
    model.eval()

    recall_lis = []
    mrr_lis = []
    with torch.no_grad():
        for seq, y, l in tqdm(dataloader_val):
            seq = seq.to(device)
            y = y.to(device)
            y_preds = model(seq, l)
            probs = F.softmax(y_preds, dim = 1)
            recall, mrr = get_metrics(probs, y, k = topk)
            recall_lis.append(recall)
            mrr_lis.append(mrr)
    
    avg_recall = np.mean(recall_lis)
    avg_mrr = np.mean(mrr_lis)
    return avg_recall, avg_mrr

data_path = './data_processed/yoochoose1_4/'
hidden = 100
emb_size = 50
ep = 100
lr = 0.001
lr_decay = 0.1
lr_step = 80
val_split = 0.99
bs = 512
topk = 20
test = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train, val, test = get_data(data_path, val_split)

train = Recsys(train)
val = Recsys(val)
test = Recsys(test)

dataloader_train = DataLoader(train, batch_size = bs, shuffle = True, collate_fn = combine)
dataloader_val = DataLoader(val, batch_size = bs, shuffle = False, collate_fn = combine)
dataloader_test = DataLoader(test, batch_size = bs, shuffle = False, collate_fn = combine)
n_items = 37484

model = MODEL(n_items, hidden, emb_size, bs).to(device)

opt = optim.Adam(model.parameters(), lr)
scheduler = StepLR(opt, step_size = lr_step, gamma = lr_decay)
loss = nn.CrossEntropyLoss()
for i in tqdm(range(ep)):
    scheduler.step(epoch = i)
    epochTrain(dataloader_train, model, opt, i, ep, loss, log_aggr = 200)

    recall, mrr = validate(dataloader_val, model)
    print('Epoch {} validation: Recall@{}: {:.2f}, MRR@{}: {:.2f} \n'.format(i, topk, recall, topk, mrr))

    ckpt_dict = {
        'state_dict': model.state_dict(),
        'epoch': i + 1,
        'optimizer': opt.state_dict()
    }

    torch.save(ckpt_dict, 'checkpoint.pth.tar')
