import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# 필요없는 행 제거
remove_list = ['팀명','순위_def', '순위_hit', '순위_pit', '순위_run', 'IP', 'W', 'L', 'G_hit', 'G_def', 'G_pit', 'G_run',
               'PA', 'AB', 'TBF', 'NP', ]
for index in remove_list:
    Data = Data.drop(index, axis = 1)


# 데이터 만들기
dataset_df = Data_cp.copy()
dataset_df = dataset_df.drop('WPCT',axis = 1)
dataset_df = dataset_df.drop('년도',axis = 1)
dataset_df = dataset_df.fillna(0)
label_df = Data['WPCT']

label = label_df.to_numpy()
dataset = dataset_df.to_numpy()
train = dataset[:166]
test = dataset[166:]
train_label = label[:166]
test_label = label[166:]

print(label.shape)
print(train.shape)
print(test.shape)

# model def
class Linear_Model(nn.Module):
    def __init__(self, feature_num, output_num):
        super(Linear_Model, self).__init__()
        self.feature_dim = feature_num
        self.output_dim = output_num
        
        self.layer = nn.Linear(self.feature_dim, self.output_dim)
    def forward(self, x):
        x = self.layer(x)
        return x

# loss, optimizer
model = Linear_Model(60,1)
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-4)


EPOCH = 10000
loss_list = []

for epoch in range(EPOCH):
    loss_sum = 0
    
    for idx in range(len(train)):
        optimizer.zero_grad()
        pred = model(torch.tensor(train[idx]).float())
        loss = loss_func(pred, torch.tensor(train_label[idx]).float())
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 50 == 0: 
        print(f"Epoch : {epoch+1}, loss : {loss_sum/len(Data)}")
    loss_list.append(loss_sum/len(Data))

plt.plot(range(EPOCH), loss_list)