import torch
import torch.nn as nn
import torch.nn.functional as F

from outer_models import trd_encoder
torch.manual_seed(1203)
from outer_models import tsf_encoder
class DMRNet(nn.Module):
    def __init__(self,voc_size,emb_dim=64,nhead=2,device='cpu',mpnn_mole=None,ddi_graph=None,ehr_graph=None,combo_matrix=None,med_vocsize=131):
        super().__init__()
        self.med_vocsize = med_vocsize
        self.device = device
        self.emb_dim = emb_dim
        self.med_dim = emb_dim*2
        # self.voc_size = voc_size
        self.voc_size = list(voc_size)
        self.nhead = 2

        self.diag_emb = nn.Embedding(voc_size[0],emb_dim,padding_idx=0,device=device)
        self.proc_emb = nn.Embedding(voc_size[1],emb_dim,padding_idx=0,device=device)

        self.dropout = nn.Dropout(p=0.2)

        self.combo_matrix = combo_matrix

        self.diag_linear_2 = nn.Sequential(*[nn.Linear(emb_dim, emb_dim * 2, device=device),
                                             nn.Tanh(),
                                             nn.Linear(emb_dim * 2, int(self.med_dim / 2), device=device),
                                             nn.Dropout(0.3)])

        self.proc_linear_2 = nn.Sequential(*[nn.Linear(emb_dim, emb_dim * 2, device=device),
                                             nn.Tanh(),
                                             nn.Linear(emb_dim * 2, int(self.med_dim / 2), device=device),
                                             nn.Dropout(0.3)])

        self.med_block = nn.Parameter(torch.randn([self.med_dim,med_vocsize],device=device))
        self.his_seq_med = nn.Parameter(torch.randn([self.med_dim,med_vocsize],device=device))
        self.combo_med = nn.Sequential(
            nn.Linear(self.med_dim,self.med_dim),
            nn.Tanh(),
            nn.Linear(self.med_dim, self.voc_size[2]-1),
            nn.Sigmoid())
        self.diag_med_block = nn.Parameter(torch.randn([self.emb_dim, med_vocsize], device=device))
        self.proc_med_block = nn.Parameter(torch.randn([self.emb_dim, med_vocsize], device=device))

        self.label_encoder = tsf_encoder(emb_dim=64,device=self.device)
        self.combo_encoder = tsf_encoder(emb_dim=64,device=self.device)

        self.diag_integ = trd_encoder(emb_dim=int(self.med_dim/2),device=device)
        self.proc_integ = trd_encoder(emb_dim=int(self.med_dim/2),device=device)

        self.combo_diag_integ = trd_encoder(emb_dim=int(self.med_dim / 2), device=device)
        self.combo_proc_integ = trd_encoder(emb_dim=int(self.med_dim / 2), device=device)
        self.diag_prob_integ = trd_encoder(emb_dim=int(med_vocsize), device=device)
        self.proc_prob_integ = trd_encoder(emb_dim=int(med_vocsize), device=device)

        self.patient_mem_contact = nn.Sequential(*[nn.Linear(self.med_dim*2,self.med_dim*2,device=device),
                                           nn.Tanh(),
                                           nn.Linear(self.med_dim*2,med_vocsize,device=device),
                                           nn.Tanh(),
                                           nn.Dropout(0.2)])

        self.his_seq_contact = nn.Sequential(*[nn.Linear(self.med_dim * 2, self.med_dim * 2, device=device),
                                                   nn.Tanh(),
                                                   nn.Linear(self.med_dim * 2, med_vocsize, device=device),
                                                   nn.Tanh(),
                                                   nn.Dropout(0.2)])

        self.combo_label_contact = nn.Sequential(*[nn.Linear(self.med_vocsize+self.combo_matrix.size()[0],
                                                             self.med_vocsize, device=device),
                                                   nn.Tanh(),
                                                   nn.Linear(self.med_vocsize, self.med_vocsize, device=device),
                                                   nn.Tanh(),
                                                   nn.Dropout(0.2)])

        self.label_combo_his_linear = nn.Sequential(*[nn.Linear(2*self.med_dim,self.med_dim, device=device),
                                                   nn.Tanh(),
                                                   nn.Linear(self.med_dim, self.med_dim, device=device),
                                                   nn.Tanh(),
                                                   nn.Dropout(0.2)])

    def history_gate_unit(self, patient_rep, all_vst_drug, contacter, all_vst_combos=None,combo_rep=None):

        all_vst_med = torch.cat([all_vst_drug,all_vst_combos],dim=-1)
        all_vst_med = self.combo_label_contact(all_vst_med)

        his_seq_mem = patient_rep[:-1]
        his_seq_mem = torch.cat([torch.zeros_like(patient_rep[0]).unsqueeze(dim=0), his_seq_mem], dim=0)
        patient_rep = self.label_combo_his_linear(torch.cat([patient_rep,combo_rep],dim=-1))

        his_seq_container = torch.zeros([patient_rep.size()[0],
                                         patient_rep.size()[0],
                                         patient_rep.size()[1] * 2],
                                        device=self.device)

        for i in range((len(patient_rep))):
            for j in range((len(his_seq_mem))):
                if j <= i:
                    his_seq_container[i, j] = torch.concat([patient_rep[i],
                                                            his_seq_mem[j]], dim=-1)
        his_seq_container = contacter(his_seq_container)

        his_seq_filter_mask = torch.tril(torch.ones([patient_rep.size()[0],
                                                     patient_rep.size()[0]], device=self.device)).unsqueeze(dim=-1)

        his_seq_enhance = his_seq_filter_mask * his_seq_container * (all_vst_drug + all_vst_med)
        his_seq_enhance = his_seq_enhance.sum(dim=1)

        return his_seq_enhance.reshape(-1, self.med_vocsize),patient_rep

    def label_2_hot(self,input,patient_rep,size):

        drug_mem = torch.nn.functional.one_hot(input.squeeze(dim=0),
                                               num_classes=size + 1).sum(dim=-2)[:, 1:].to(torch.float32)

        drug_mem_pad = torch.zeros_like(drug_mem[0]).unsqueeze(dim=0)

        drug_mem = torch.cat([drug_mem_pad, drug_mem], dim=0)[:drug_mem.size()[0]].unsqueeze(dim=0).repeat([
            patient_rep.size()[0], 1, 1]).to(self.device)
        return drug_mem

    def encode_patient_label(self,diag_seq,proc_seq):

        patient_rep = torch.concat([self.diag_integ(diag_seq),
                                    self.proc_integ(proc_seq)], dim=-1)

        return patient_rep

    def encode_patient_combo(self,combo_diag,combo_proc):
        combo_rep = torch.concat([self.combo_diag_integ(combo_diag),
                                  self.combo_proc_integ(combo_proc)], dim=-1)

        return combo_rep

    def seq_view_score_calculate(self,patient_rep,drug_mem,combo_rep,combos):

        patient_rep = patient_rep.squeeze(dim=1)
        his_enhance = self.history_gate_unit(patient_rep, drug_mem, self.patient_mem_contact,
                                             all_vst_combos=combos, combo_rep=combo_rep)
        medication_score = patient_rep @ self.med_block
        return medication_score + his_enhance

    def token_view_score_calculate(self,diag,proc,diag_med_block,proc_med_block):

        diag_probseq = diag @ diag_med_block
        proc_probseq = proc @ proc_med_block

        diag_prob = self.diag_prob_integ(diag_probseq)
        proc_prob = self.proc_prob_integ(proc_probseq)

        token_view_score = (diag_prob + proc_prob).squeeze()
        return token_view_score

    def encoder(self,diag,proc,diag_mask=None,proc_mask=None):

        diag = self.diag_emb(diag)
        proc = self.proc_emb(proc)

        diag_seq,proc_seq = self.label_encoder(diag,proc,diag_mask,proc_mask)
        combo_diag,combo_proc = self.combo_encoder(diag,proc,diag_mask,proc_mask)

        label_patient_rep = self.encode_patient_label(diag,proc)
        combo_patient_rep = self.encode_patient_combo(combo_diag,combo_proc)

        diag_seq = self.diag_linear_2(diag_seq)
        proc_seq = self.proc_linear_2(proc_seq)

        return diag_seq,proc_seq,label_patient_rep,combo_patient_rep.reshape(-1,self.med_dim)

    def decoder(self,diag,proc,drug_mem=None,patient_rep=None,combo_rep=None,combos=None):

        drug_mem = self.label_2_hot(drug_mem,patient_rep,size=self.med_vocsize)
        combos = self.label_2_hot(combos,patient_rep,size=self.combo_matrix.size()[0])

        #+++++++++++++++++++++++++++++++++++++++++++++
        seq_view_score = self.seq_view_score_calculate(patient_rep=patient_rep,drug_mem=drug_mem,combo_rep=combo_rep,
                                                       combos=combos)
        #============================================================
        token_view_score = self.token_view_score_calculate(diag,proc,self.diag_med_block,self.proc_med_block)
        # ==============================================

        prob = seq_view_score + token_view_score

        combo_prob = self.combo_med(combo_rep)
        prob = F.sigmoid(prob)

        prob = prob.reshape(-1, self.med_vocsize)
        combo_prob = combo_prob.reshape(-1, self.voc_size[2]-1)

        prob_padder = torch.full_like(prob.T[0], 0).unsqueeze(dim=0).T
        prob = torch.cat([prob_padder, prob], dim=-1)
        combo_prob = torch.cat([prob_padder,combo_prob],dim=-1)

        return prob,prob*prob.T.unsqueeze(dim=-1),combo_prob


    def forward(self,input):

        diag_hid,proc_hid,patient_rep,combo_rep = self.encoder(input[0],input[1],input[3],input[4])

        decoder_out = self.decoder(diag_hid,proc_hid,input[5],input[6],input[2],
                                   patient_rep=patient_rep,combo_rep=combo_rep,combos=input[7])

        return decoder_out

