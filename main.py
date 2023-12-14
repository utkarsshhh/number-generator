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

features = data.drop(labels = 'result',axis = 1)
target = data['result']

x = torch.tensor(features.values,dtype=torch.float32)
y = torch.tensor(target.values, dtype=torch.float32)

dataset = TensorDataset(x,y)

train_size = int(0.8*len(dataset))
test_size = len(dataset)-train_size

train_data,test_data = random_split(dataset,[train_size,test_size])

batch_size = 64

train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size = batch_size,shuffle = False)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Linear(8, 1)
        )
    def forward(self,x):
        return self.model(x)


model = MyModel()
optimizer = SGD(model.parameters(),lr = 0.001)
criterion  = nn.MSELoss()
n_epochs = 30

for i in range(n_epochs):
    model.train()
    for features,target in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.view(-1),target)
        loss.backward()
        optimizer.step()

    print ('Loss for Epoch ',i+1," : ",loss)

model.eval()
for features,target in test_loader:
    outputs = model(features)
    loss =criterion(outputs.view(-1),target)

    print ()

