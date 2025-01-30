# coding: utf-8
# 2021/3/23 @ tongshiwei
import sys
sys.path.append("/team_code/JR/CDM/EduCDM-main/")

import logging
from EduCDM import GDDINA
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

data_name="elion"
if data_name == "mathdial":
    train_data = pd.read_csv("../../../data/mathdial/train.csv")
    valid_data = pd.read_csv("../../../data/mathdial/valid.csv")
    test_data = pd.read_csv("../../../data/mathdial/test.csv")
    item_data = pd.read_csv("../../../data/mathdial/item.csv")
    knowledge_num = 147

if data_name == "comta":
    train_data = pd.read_csv("../../../data/IRE_Comta/train.csv")
    valid_data = pd.read_csv("../../../data/IRE_Comta/valid.csv")
    test_data = pd.read_csv("../../../data/IRE_Comta/test.csv")
    item_data = pd.read_csv("../../../data/IRE_Comta/item.csv")
    knowledge_num = 165

if data_name == "elion":
    train_data = pd.read_csv("../../../data/elion/train_elion.csv")
    valid_data = pd.read_csv("../../../data/elion/valid_elion.csv")
    test_data = pd.read_csv("../../../data/elion/test_elion.csv")
    item_data = pd.read_csv("../../../data/elion/item.csv")
    knowledge_num = 158



def code2vector(x):
    vector = [0] * knowledge_num
    for k in eval(x):
        vector[k - 1] = 1
    return vector


item_data["knowledge"] = item_data["knowledge_code"].apply(code2vector)
item_data.drop(columns=["knowledge_code"], inplace=True)

train_data = pd.merge(train_data, item_data, on="item_id")
valid_data = pd.merge(valid_data, item_data, on="item_id")
test_data = pd.merge(test_data, item_data, on="item_id")

batch_size = 32


def transform(x, y, z, k, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(k, dtype=torch.float32),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


train, valid, test = [
    transform(data["user_id"], data["item_id"], data["score"], data["knowledge"], batch_size)
    for data in [train_data, valid_data, test_data]
]
auc_all=0
acc_all=0
for i in range(5):
    logging.getLogger().setLevel(logging.INFO)

    cdm = GDDINA(4164, 17747, knowledge_num)

    cdm.train(train, valid, epoch=100)
    cdm.save("dina.params")

    cdm.load("dina.params")
    auc, accuracy = cdm.eval(test)
    auc_all += auc
    acc_all += accuracy
auc_ave = auc_all / 5
acc_ave = acc_all / 5
print(data_name)
print("auc_ave: %.6f, accuracy_ave: %.6f" % (auc_ave, acc_ave))
