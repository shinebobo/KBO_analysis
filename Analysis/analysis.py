import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim 

# Load data

Data = pd.read_csv("Data.csv")

# 필요없는 행 제거
remove_list = ['팀명','순위_def', '순위_hit', '순위_pit', '순위_run', 'IP', 'W', 'L', 'G_hit', 'G_def', 'G_pit', 'G_run',
               'PA', 'AB', 'TBF', 'NP', ]

for index in remove_list:
    Data = Data.drop(index, axis = 1)

use_list = ['WPCT', '년도', 'H_hit', '2B_hit', '3B_hit', 'HR_hit', 'BB_hit', 'SF_hit','R_hit', 'H_pit', 
            '2B_pit', '3B_pit', 'HR_pit', 'BB_pit', 'SF_pit', 'R_pit']

Data = Data[use_list]
Data['H_hit'] = Data['H_hit'] - Data['2B_hit'] - Data['3B_hit'] - Data['HR_hit']
Data['H_pit'] = Data['H_pit'] - Data['2B_pit'] - Data['3B_pit'] - Data['HR_pit']

# get R_hit & R_run
HIT_SITUATION = ['H_hit', '2B_hit', '3B_hit', 'HR_hit', 'BB_hit', 'SF_hit']
# total base case (from 14 to 21)
BASE_SCORE = {'H_hit':[0,0,0,0,1,1,1,1], '2B_hit':[0,0,1,1,1,1,2,2], '3B_hit':[0,1,1,1,2,2,2,3], 
              'HR_hit':[1,2,2,2,3,3,3,4], 'BB_hit':[0,0,0,0,0,0,0,1], 'SF_hit':[0,0,0,0,1,1,1,1]}
BASE_CASE_P = [230992, 85018, 37902, 36455, 11880, 13146, 10996, 13890]
BASE_CASE_P = [x / sum(BASE_CASE_P) for x in BASE_CASE_P]
p_hit = {'H_hit':0.6, '2B_hit':0.7 , '3B_hit':0.5, 'HR_hit':1, 'BB_hit':1, 'SF_hit':0.3}
# p_hit = {'H_hit':1, '2B_hit':1 , '3B_hit':1, 'HR_hit':1, 'BB_hit':1, 'SF_hit':1}

p_run = {'H_hit':0.4, '2B_hit':0.3 , '3B_hit':0.5, 'HR_hit':0, 'BB_hit':0, 'SF_hit':0.7}
# p_run = {'H_hit':0, '2B_hit':0 , '3B_hit':0, 'HR_hit':0, 'BB_hit':0, 'SF_hit':0}

R_hit_list = []
R_run_list = []

for idx in Data.index:
    R_hit = 0
    R_run = 0
    for hit_case in HIT_SITUATION:
        for base_p, base_sc in zip(BASE_CASE_P, BASE_SCORE[hit_case]):
            R_hit = R_hit + Data[hit_case][idx] * p_hit[hit_case] * base_sc * base_p
            R_run = R_run + Data[hit_case][idx] * p_run[hit_case] * base_sc * base_p
    
    R_hit_list.append(R_hit)
    R_run_list.append(R_run)


Data['R_hit_sc'] = R_hit_list
Data['R_run_sc'] = R_run_list

# get R_pit & R_def
PIT_SITUATION = ['H_pit', '2B_pit', '3B_pit', 'HR_pit', 'BB_pit', 'SF_pit']
# total base case (from 14 to 21)
BASE_SCORE = {'H_pit':[0,0,0,0,1,1,1,1], '2B_pit':[0,0,1,1,1,1,2,2], '3B_pit':[0,1,1,1,2,2,2,3], 
              'HR_pit':[1,2,2,2,3,3,3,4], 'BB_pit':[0,0,0,0,0,0,0,1], 'SF_pit':[0,0,0,0,1,1,1,1]}
BASE_CASE_P = [230992, 85018, 37902, 36455, 11880, 13146, 10996, 13890]
BASE_CASE_P = [x / sum(BASE_CASE_P) for x in BASE_CASE_P]
p_pit = {'H_pit':0.6, '2B_pit':0.7 , '3B_pit':0.5, 'HR_pit':1, 'BB_pit':1, 'SF_pit':0.7}
# p_pit = {'H_pit':1, '2B_pit':1 , '3B_pit':1, 'HR_pit':1, 'BB_pit':1, 'SF_pit': 1}

p_def = {'H_pit':0.4, '2B_pit':0.3 , '3B_pit':0.5, 'HR_pit':0, 'BB_pit':0, 'SF_pit':0.3}
# p_def = {'H_pit':0, '2B_pit':0 , '3B_pit':0, 'HR_pit':0, 'BB_pit':0, 'SF_pit': 0}

R_pit_list = []
R_def_list = []

for idx in Data.index:
    R_pit = 0
    R_def = 0
    for pit_case in PIT_SITUATION:
        for base_p, base_sc in zip(BASE_CASE_P, BASE_SCORE[pit_case]):
            R_pit = R_pit + Data[pit_case][idx] * p_pit[pit_case] * base_sc * base_p
            R_def = R_def + Data[pit_case][idx] * p_def[pit_case] * base_sc * base_p
    
    R_pit_list.append(R_pit)
#     R_def_list.append(Data['R_pit'][idx] - R_pit)
    R_def_list.append(R_def)

Data['R_pit_sc'] = R_pit_list
Data['R_def_sc'] = R_def_list

# normalization
Data_cp = Data.copy()
# for year in range(2001, 2022):  
# print(df.dtypes)
# for col, type in zip(df, df.dtypes):
#     print(col, type)

for year in range(2001, 2022):
    df = Data[Data.년도 == year]
    normalization_df = (df - df.mean())/df.std()
    Data_cp[Data.년도 == year] = normalization_df
    
    
dataset_df = Data_cp.copy()
dataset_df = dataset_df.drop('WPCT',axis = 1)
dataset_df = dataset_df.drop('년도',axis = 1)
dataset_df = dataset_df.fillna(0)
label_df = Data['WPCT']

label = label_df.to_numpy()
# dataset = dataset_df.to_numpy()
dataset = dataset_df[['R_hit_sc', 'R_pit_sc', 'R_run_sc', 'R_def_sc']].to_numpy()
# dataset = dataset_df[['R_hit', 'R_pit']].to_numpy()


train = dataset
train_label = label


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
model = Linear_Model(4,1)
# model = Linear_Model(2,1)
loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-4)


EPOCH = 500
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
plt.xlabel('Epoch')
plt.ylabel('Loss')



# params = list(model.parameters())
# print(params[0][0])

# names = ['R_hit', 'R_pit', 'R_run', 'R_def']
# for name, param in zip(names,params[0][0]):
#     print(f"Name: {name}, weight: {param}")

# print(f"Name: Bias, weight: {params[1][0]}")   