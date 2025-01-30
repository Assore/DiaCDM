# coding: utf-8
# 2023/7/3 @ WangFei

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from EduCDM import CDM
from torch_geometric.nn import GCNConv, global_mean_pool
import json
import csv
import os



class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
    def forward(self, x):
        # 初始化隐藏状态和记忆细胞
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # 形状变为 [batch_size, hidden_size]

        # 全连接层映射到输出维度
        out = self.fc(out)  # 形状变为 [batch_size, output_size]
        return out

class GNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(16, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.linear = torch.nn.Linear(input_dim, 16)

    def forward(self, x, edge_index, batch):
        x=self.linear(x)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x



class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)



class Net(nn.Module):

    def __init__(self, exer_n, student_n, knowledge_n, mf_type, dim,KCs):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable
        self.KCs = KCs.to('cuda:2')

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, 1)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.2)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.2)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        self.teacher_diff = GNNEncoder(input_dim=4096, hidden_dim=self.emb_dim*2, output_dim=self.emb_dim)
        self.teacher_disc = GNNEncoder(input_dim=4096, hidden_dim=self.emb_dim * 2, output_dim=1)
        self.student=nn.Linear(4096,self.emb_dim)
        self.evaluate=nn.Linear(4096,self.emb_dim)

        self.attention=nn.MultiheadAttention(embed_dim=20, num_heads=2,batch_first=True)
        self.KC_emb=nn.Linear(4096,self.emb_dim)

        self.a=SimpleLSTM(4096, 128, 2, 1)
        self.b = SimpleLSTM(4096, 128, 2, 1)
        self.c = SimpleLSTM(4096, 128, 2, 1)


    def forward(self, stu_id, input_exercise, input_knowledge_point,teacher_text,teacher,student,evaluate,visual=False):
        # before prednet
        stu_emb = self.student_emb(stu_id)

        theta = torch.squeeze(self.student_emb(stu_id), dim=-1)
        a = torch.squeeze(self.a(teacher_text.unsqueeze(1)), dim=-1)
        b = torch.squeeze(self.b(teacher_text.unsqueeze(1)), dim=-1)
        c = torch.squeeze(self.c(teacher_text.unsqueeze(1)), dim=-1)
        c = torch.sigmoid(c)

        D = 1.702


        # get knowledge proficiency

        a = F.softplus(a)
        if visual==True:
            stu_emb=torch.sigmoid(stu_emb.sum(dim=-1, keepdim=False))
        return c + (1 - c) / (1 + torch.exp(-D * a * (theta - b)))



class KaNCD(CDM):
    def __init__(self, **kwargs):
        super(KaNCD, self).__init__()
        mf_type = kwargs['mf_type'] if 'mf_type' in kwargs else 'gmf'
        self.net = Net(kwargs['exer_n'], kwargs['student_n'], kwargs['knowledge_n'], mf_type, kwargs['dim'],kwargs['KCs'])



    def train(self, train_set, train_graph,valid_set,valid_graph, lr=0.001, device='cpu', epoch_n=15):
        logging.info("traing... (lr={})".format(lr))
        self.net = self.net.to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        for epoch_i in range(epoch_n):
            self.net.train()
            epoch_losses = []
            batch_count = 0

            for batch_data,batch_graph in zip(train_set,train_graph):
                batch_count += 1
                user_info, item_info, knowledge_emb, teacher_text,student, evalu, y = batch_data
                user_info: torch.Tensor = user_info.to(device)
                item_info: torch.Tensor = item_info.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                teacher_text: torch.Tensor = teacher_text.to(device)
                student: torch.Tensor = student.to(device)
                evalu: torch.Tensor = evalu.to(device)
                teacher: torch.Tensor = batch_graph.to(device)
                y: torch.Tensor = y.to(device)
                pred = self.net(user_info, item_info, knowledge_emb,teacher_text,teacher,student,evalu)

                loss = loss_function(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())
            # self.processData("IRE_Comta",epoch_i,"train", train_set, train_graph)
            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            logging.info("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            auc, acc = self.eval(valid_set, valid_graph)
            print("[Epoch %d] auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))
            logging.info("[Epoch %d] auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))

        return auc, acc

    def eval(self, test_data,test_graph, device="cuda:2",visual=False,output_file="./NONE.csv"):
        logging.info('eval ... ')
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        output_file = output_file
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入表头
            header = ["student_id", "item_id", "Q_matrix", "threeLevelSum","stuState","queMatch", "staInRes", "staInTea", "diff", "discri", "correct","pred"]
            writer.writerow(header)
            for batch_data,batch_graph in zip(test_data,test_graph):
                user_info, item_info, knowledge_emb, teacher_text,student, evalu, y = batch_data
                user_info: torch.Tensor = user_info.to(device)
                item_info: torch.Tensor = item_info.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                teacher_text: torch.Tensor = teacher_text.to(device)
                student: torch.Tensor = student.to(device)
                evalu: torch.Tensor = evalu.to(device)
                teacher: torch.Tensor = batch_graph.to(device)
                y: torch.Tensor = y.to(device)
                pred = self.net(user_info, item_info, knowledge_emb, teacher_text,teacher, student, evalu)

                if visual:
                    pred,stu,item,Q_matrix,t,init,t_e,t_s,s_e,diff,evaluate = self.net(user_info, item_info, knowledge_emb, teacher_text, teacher, student, evalu,visual=True)
                    correct=y
                    pred_1 = [round(x) for x in pred.cpu().tolist()]
                    stu = stu.cpu().tolist()
                    item = item.cpu().tolist()
                    Q_matrix = Q_matrix.cpu().tolist()
                    t = t.cpu().tolist()
                    init = init.cpu().tolist()
                    t_e = t_e.cpu().tolist()
                    t_s = t_s.cpu().tolist()
                    s_e = s_e.cpu().tolist()
                    diff = diff.cpu().tolist()
                    evaluate = evaluate.cpu().tolist()
                    correct = correct.cpu().tolist()

                    for j in range(len(stu)):
                        row = [
                            stu[j],
                            item[j],
                            list(Q_matrix[j]),
                            list(t[j]),
                            list(init[j]),
                            list(t_e[j]),
                            list(t_s[j]),
                            list(s_e[j]),
                            list(diff[j]),
                            list(evaluate[j]),
                            correct[j],
                            pred_1[j]
                        ]
                        writer.writerow(row)


                y_pred.extend(pred.detach().cpu().tolist())
                y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)

    def processData(self,dataname,epoch,type,test_data,test_graph):

        ep=str(epoch)
        file_dir="../../data/"+dataname
        file_name=file_dir+'/'+type+'_ep'+ep+'.csv'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        self.eval(test_data,test_graph,visual=True,output_file=file_name)
        # students= torch.arange(0, 123,device='cuda')
        # stu_emb = self.net.student_emb(students)  # shape: (123, embedding_dim)
        #
        # # 3. 变换形状并重复
        # stu_emb = stu_emb.view(123, 1, self.net.emb_dim).repeat(1, self.net.knowledge_n, 1)
        # # shape: (batch_size, knowledge_n, embedding_dim)
        #
        # # 4. 沿最后一个维度求和并应用 sigmoid
        # stu_emb = torch.sigmoid(stu_emb.sum(dim=-1, keepdim=False))
        # # shape: (batch_size, knowledge_n)
        # stu_emb_cpu = stu_emb.cpu().tolist()
        # # 5. 将每一行保存为 CSV 文件中的一个列表
        # output_csv = file_name
        # with open(output_csv, mode="w", newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["stusta"])  # 列名
        #     for row in stu_emb.tolist():
        #         writer.writerow([row])  # 每一行是一个 list



