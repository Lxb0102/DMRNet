import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from data_loader import pad_batch_v2_train, pad_batch_v2_eval
import numpy as np
from util import llprint
import dill
from outer_models import multi_label_metric,ddi_rate_score
from model_net import DMRNet

torch.manual_seed(1203)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def main(temp_number=None):
    device = "cuda"
    mimic_ver='iv'
    voc = dill.load(open(r'voc_final_{}.pkl'.format(mimic_ver), 'rb'))
    combo_dict = dill.load(open(r'data'+mimic_ver+'/basic_combos.pkl', 'rb'))
    # print(combo_dict)
    print(len(combo_dict))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (len(diag_voc.idx2word) + 1, len(pro_voc.idx2word) + 1, len(combo_dict) + 1)

    combo_label_convert = torch.zeros(len(combo_dict)+1,voc_size[2]+1,device=device)
    for i in range(1,len(combo_dict)):
        combo_label_convert[i][combo_dict[i]] = 1
    dill.dump(combo_label_convert, open(r'data/combo_label_convert_matrix.pkl', 'wb'))
    # return None
    # <
    combo_label_convert = dill.load(open(r'data/combo_label_convert_matrix.pkl', 'rb')).to(device)
    data = dill.load(open(r'records_final_{}.pkl'.format(mimic_ver), 'rb'))#[::100]

    for patient in range(len(data)):
        for vst in range(len(data[patient])):

            data[patient][vst][0]=[i+1 for i in data[patient][vst][0]]
            data[patient][vst][1]=[i+1 for i in data[patient][vst][1]]
            data[patient][vst][2]=[i+1 for i in data[patient][vst][2]]

    combo_drug = dill.load(open(r'data/'+mimic_ver+'/pattern_records_final.pkl','rb'))
    vst_cnt = 0
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]

    for patient in range(len(data)):
        for vst in range(len(data[patient])):
            data[patient][vst].insert(3,combo_drug[vst_cnt])
            vst_cnt+=1

    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    ddi_matrix = dill.load(open(r'ddi_A_final.pkl','rb'))
    ddi_matrix = torch.tensor(ddi_matrix,device=device)

    med_vocsize=len(med_voc.idx2word)

    print(voc_size)

    train_loader = DataLoader(data_train, batch_size=1, collate_fn=pad_batch_v2_train, shuffle=True, pin_memory=False)
    eval_loader = DataLoader(data_eval, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=True, pin_memory=False)
    test_loader = DataLoader(data_test, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=True, pin_memory=False)

    model = DMRNet(emb_dim=64, voc_size=voc_size, med_vocsize=med_vocsize,device=device, ddi_graph=ddi_matrix,combo_matrix=combo_label_convert).to(device)

    model.load_state_dict(torch.load(r'state_dict\{}\best.pt'.format(mimic_ver)))

    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), lr=0.0001)
    EPOCH = 1
    bce_loss = nn.BCELoss()

    for epoch in range(EPOCH):
        ddi_rate = 0
        avg_precise = 0
        avg_recall = 0
        avg_f1 = 0
        count = 1e-6
        model.to(device)
        model.train()
        model_train = False
        all_his = True

        if model_train:
            for index,datas in enumerate(train_loader):

                datas = [i.to(device) for i in datas]
                output = list(model(datas))

                gt_container = torch.zeros_like(output[0], device=device).reshape(-1,med_vocsize+1)
                combo_container = torch.zeros_like(output[2], device=device).reshape(-1,voc_size[2])
                loss3_target = np.full((output[2].size()), -1).reshape([-1,voc_size[2]])

                if all_his:
                    for batch_idx, batch in enumerate(datas[2][0]):
                        for idx, seq in enumerate(batch):
                            gt_container[batch_idx][seq] = 1.0
                    for batch_idx, batch in enumerate(datas[7][0]):
                        for idx, seq in enumerate(batch):

                            combo_container[batch_idx][seq] = 1.0

                else:
                    gt_container[0][datas[2][0][-1]] = 1

                if all_his:
                    gt_container[:,0] = 0
                    combo_container[:,0]=0
                else:
                    gt_container[0] = 0

                loss_1 = bce_loss(output[0],gt_container)
                loss_3 = bce_loss(output[2], combo_container)

                loss = (loss_1 + 0.5*loss_3)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                llprint('\r|'+\
                         '#'*int(50*(index/len(train_loader)))+\
                         '-'*int(50-50*(index/len(train_loader)))+\
                         '|{:.2f}%|train_step:{}/{}'.format(100*(index/len(train_loader)),index,len(train_loader))
                        )

        print()

        model.eval()
        prob_container = []
        gt_container = []
        labels_container = []
        ddi_cnt = 0
        ddi_all_cnt = 0
        avg_med = 0
        for index, datas in enumerate(eval_loader):
            datas = [i.to(device) for i in datas]
            output = model(datas)[0]

            gt_data = datas[2][0]

            for idx,vst in enumerate(output.reshape(-1,med_vocsize+1)):
                gt_temp = torch.zeros_like(vst, device=device)
                gt_temp[gt_data[idx]] = 1
                gt_temp[0] = 0
                avg_med += vst.sum()
                out_labels = torch.where(vst > 0.35, 1.0, 0.0)

                ddi_temp_container = out_labels*out_labels.T.unsqueeze(dim=-1)
                labels_container.append(out_labels)
                prob_container.append(vst)
                gt_container.append(gt_temp)

                if gt_temp.sum()!=0:
                    precise = (out_labels * gt_temp).sum() / (out_labels.sum() + 1e-9)
                    recall = (out_labels * gt_temp).sum() / (gt_temp.sum() + 1e-9)
                else:
                    continue
                avg_precise += precise
                avg_recall += recall

                if (precise + recall) == 0:
                    count += 1
                    continue
                else:
                    f1 = (2.0 * precise * recall) / (precise + recall)
                avg_f1 += f1

                ddi_cnt += ddi_rate_score(ddi_temp_container,ddi_matrix)[0]
                ddi_all_cnt += ddi_rate_score(ddi_temp_container,ddi_matrix)[1]

                count += 1

            llprint('\r|' + \
                    '@' * int(50 * (index / len(eval_loader))) + \
                    '-' * int(50 - 50 * (index / len(eval_loader))) + \
                    '|{:.2f}%|eval_step:{}/{}'.format(100 * (index / len(eval_loader)), index, len(eval_loader))
                    )
        avg_precise=avg_precise/count
        avg_recall=avg_recall/count
        avg_f1=avg_f1/count

        jac,prauc,F_1 = multi_label_metric(gt_container,labels_container,prob_container,voc_size=voc_size)
        try:
            ddi_rate = ddi_cnt/ddi_all_cnt
        except:
            print('没有药物相互作用对')
            ddi_rate = 0

        print('\navg_prc = {}\n'.format(avg_precise),
              'avg_rec = {}\n'.format(avg_recall),
              'jac = {}\n'.format(jac),
              'prauc = {}\n'.format(prauc),
              'avg_f1 = {}\n'.format(avg_f1),
              'ddi_rate = {}\n'.format(ddi_rate),
              'avg_med = {}\n'.format(avg_med/count)
               )

        print(f'epoch{epoch}\n')


# for i in range(1):
for i in [1]:
    main(i)
