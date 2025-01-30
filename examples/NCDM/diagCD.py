# coding: utf-8
# 2021/4/1 @ WangFei
import sys

from sentry_sdk.utils import epoch

sys.path.append("/team_code/JR/CDM/EduCDM-main/")
import logging
from EduCDM.DiagNCDM import NCDM
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os

tqdm.pandas()

model_name="/team_code/JR/Meta-Llama-3.1-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

data_name="comta"
if data_name == "mathdial":
    train_data = pd.read_csv("../../data/mathdial/train.csv")
    valid_data = pd.read_csv("../../data/mathdial/valid.csv")
    test_data = pd.read_csv("../../data/mathdial/test.csv")
    df_item = pd.read_csv("../../data/mathdial/item.csv")
if data_name == "comta":
    train_data = pd.read_csv("../../data/comta/train.csv")
    valid_data = pd.read_csv("../../data/comta/valid.csv")
    test_data = pd.read_csv("../../data/comta/test.csv")
    df_item = pd.read_csv("../../data/comta/item.csv")

item2knowledge = {}
knowledge_set = set()
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)

batch_size = 32
user_n = np.max(train_data['user_id'])
item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])
knowledge_n = np.max(list(knowledge_set))

def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)  # 分词
    with torch.no_grad():
        outputs = model(**inputs)  # 前向传播
    # 提取句子级别的表示（平均池化）
    sentence_embedding = outputs.last_hidden_state.mean(dim=1)  # (1, hidden_size)
    return sentence_embedding.squeeze(0).numpy()  # 转为 numpy 数组

def transform(type,user, item, item2knowledge,teacher, score, batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0
    file_path='../../data/'+data_name+'/'+type+'_teacher.npy'
    if not os.path.exists(file_path):
        encoded_series = teacher.progress_apply(lambda x: encode_text(x, tokenizer, model))
        teacher_tensor = torch.tensor(encoded_series, dtype=torch.float32)
        numpy_array = teacher_tensor.numpy()  # 转换为 numpy 数组
        np.save(file_path, numpy_array)
    else:
        loaded_numpy_array = np.load(file_path)  # 从 .npy 文件加载
        teacher_tensor = torch.from_numpy(loaded_numpy_array)

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        teacher_tensor,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

train_set=transform("train",train_data["user_id"], train_data["item_id"], item2knowledge, train_data["teacher"], train_data["score"], batch_size)
valid_set=transform("valid",valid_data["user_id"], valid_data["item_id"], item2knowledge, valid_data["teacher"], valid_data["score"], batch_size)
test_set=transform("test",test_data["user_id"], test_data["item_id"], item2knowledge, test_data["teacher"], test_data["score"], batch_size)
auc_all=0
acc_all=0
for i in range(5):
    logging.getLogger().setLevel(logging.INFO)
    cdm = NCDM(knowledge_n, item_n, user_n)
    cdm.train(train_set, valid_set, epoch=200, device="cuda")
    cdm.save("ncdm.snapshot")
    cdm.load("ncdm.snapshot")
    auc, accuracy = cdm.eval(test_set)
    auc_all+=auc
    acc_all+=accuracy
    print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
auc_ave=auc_all/5
acc_ave=acc_all/5
print(data_name)
print("auc_ave: %.6f, accuracy_ave: %.6f" % (auc_ave, acc_ave))


