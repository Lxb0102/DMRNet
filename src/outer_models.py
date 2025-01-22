import torch
import torch.nn as nn
import numpy as np
import random
device = 'cuda:0'
from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    average_precision_score,
    )
from itertools import chain

import dill
import torch.nn.functional as F

mimic_ver = 'iv'
voc = dill.load(open(r'voc_final_{}.pkl'.format(mimic_ver), 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

class tsf_encoder(nn.Module):
    def __init__(self, emb_dim=64, nhead=2, device='cpu'):
        super().__init__()
        self.emb_dim=emb_dim
        self.nhead = nhead
        self.diag_linear_1 = nn.Sequential(*[nn.Linear(emb_dim, emb_dim * 2, device=device),
                                             nn.Tanh(),
                                             nn.Linear(emb_dim * 2, emb_dim, device=device),
                                             nn.Dropout(0.3)])

        self.proc_linear_1 = nn.Sequential(*[nn.Linear(emb_dim, emb_dim * 2, device=device),
                                             nn.Tanh(),
                                             nn.Linear(emb_dim * 2, emb_dim, device=device),
                                             nn.Dropout(0.3)])
        self.diag_encoder = nn.TransformerEncoderLayer(emb_dim, nhead, batch_first=True, dropout=0.2,
                                                       device=device)
        self.proc_encoder = nn.TransformerEncoderLayer(emb_dim, nhead, batch_first=True, dropout=0.2,
                                                       device=device)

    def forward(self,diag_seq,proc_seq,diag_mask=None,proc_mask=None):
        max_diag_num = diag_seq.size()[-2]
        max_proc_num = proc_seq.size()[-2]
        max_visit_num = diag_seq.size()[1]
        batch_size = 1
        # print(diag_seq.size())
        diag_seq = self.diag_linear_1(diag_seq.view(batch_size * max_visit_num,
                                                                   max_diag_num, self.emb_dim))
        proc_seq = self.proc_linear_1(proc_seq.view(batch_size * max_visit_num,
                                                                   max_proc_num, self.emb_dim))

        d_mask_matrix = diag_mask.view(batch_size * max_visit_num, max_diag_num).unsqueeze(dim=1).unsqueeze(
            dim=1).repeat(1, self.nhead, max_diag_num, 1)  # [batch*seq, nhead, input_length, output_length]
        d_mask_matrix = d_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_diag_num, max_diag_num)

        p_mask_matrix = proc_mask.view(batch_size * max_visit_num, max_proc_num).unsqueeze(dim=1).unsqueeze(
            dim=1).repeat(1, self.nhead, max_proc_num, 1)
        p_mask_matrix = p_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_proc_num, max_proc_num)

        diag_seq = self.diag_encoder(diag_seq, src_mask=d_mask_matrix).view(-1, max_diag_num,
                                                                            self.emb_dim)
        proc_seq = self.proc_encoder(proc_seq, src_mask=p_mask_matrix).view(-1, max_proc_num,
                                                                            self.emb_dim)
        return diag_seq,proc_seq

class trd_encoder(nn.Module):
    def __init__(self,emb_dim=64,out_dim=None,device='cpu'):
        super().__init__()
        if not out_dim:
            out_dim = emb_dim

        self.f_encoder = nn.GRU(emb_dim, out_dim, batch_first=True, device=device,dropout=0.2)

    def forward(self,input):

        f_out = self.f_encoder(input)[1].transpose(0,1)
        r_out = self.f_encoder(torch.flip(input,dims=[1]))[1].transpose(0, 1)

        return f_out+r_out

def ddi_rate_score(record, path=None):
    # return 0
    # ddi rate
    ddi_A = path
    all_cnt = 0
    dd_cnt = 0
    tril_matrix = torch.tril(torch.ones(132,132).to(record[0]))
    tril_matrix -= torch.eye(132,device=device)

    for output in record:
        dd_cnt += (output*ddi_A*tril_matrix).sum().sum()
        all_cnt += (output*tril_matrix).sum().sum()

    if all_cnt == 0:
        return 0,1
    return dd_cnt, all_cnt

def multi_label_metric(y_gt, y_pred, y_prob,voc_size=None):
    # return 0,0
    voc_size = list(voc_size)
    voc_size[2] = len(med_voc.idx2word)+1
    y_gt = torch.concat(y_gt).reshape(-1,voc_size[2]).cpu().detach().numpy()
    y_pred = torch.cat(y_pred).cpu().reshape(-1,voc_size[2]).detach().numpy()
    y_prob = torch.cat(y_prob).cpu().reshape(-1,voc_size[2]).detach().numpy()

    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if len(union) == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score


    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average="macro"))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average="macro"))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(
                average_precision_score(y_gt[b], y_prob[b], average="macro")
            )
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)


    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    F1 = f1(y_gt,y_pred)

    return ja, prauc ,F1
