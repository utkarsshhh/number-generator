# import random
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset,random_split




# Generate the dataset
# dataset = []
# num_samples = 3000
#
# for _ in range(num_samples):
#     num1 = random.randint(-10, 10)
#     num2 = random.randint(-10, 10)
#
#     result = num1 + num2
#
#     dataset.append({'num1': num1, 'num2': num2, 'result': result})
#
# data  = pd.DataFrame(dataset)
# data.to_csv('data.csv',index=False)

data = pd.read_csv('data.csv')
print (data.head())

features = data.drop(lables = 'result',axis = 1)
target = data['result']

x = torch.tensor(features.values)
y = torch.tensor(target.values)

dataset = TensorDataset(x,y)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2,16),
            nn.BatchNorm1d(),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.BatchNorm1d(),
            nn.ReLU(),

            nn.Linear(8, 1)
        )
    def forward(self,x):
        return self.model(x)

optimizer = SGD()
loss  = nn.MSELoss()
n_epochs = 20

for _ in range(n_epochs):
