import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, layer_num):
        super(MLP, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_dim)
        self.layers = nn.ModuleList([Feed_Forward_Module(hidden_dim) for _ in range(layer_num)])
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, input1):
        embedded_input = self.embedding(input1)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        out = self.linear(out)
        prob = F.softmax(out)
        
        return prob


class Feed_Forward_Module(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input1):
        return F.relu(self.W(input1))



class DNN():
    def __init__(self, input_size=3000, output_size=2):
        self.output_size = output_size
        self.model = MLP(input_size=input_size, output_size=output_size, hidden_dim=128, layer_num=3)
    
    def train(self, X, y, batch_size=128, epoch=20, lr=1e-3):
        data = X
        label =  torch.eye(self.output_size)[y]   

        train_dataset = TensorDataset(torch.Tensor(data), torch.Tensor(label))
        iter = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for _ in range(epoch):
            for X, y in iter:
                y_pred = self.model(X)

                loss = ((y-y_pred)**2).mean()
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                # print(loss)

    def predict(self, X):
        y = torch.argmax(self.model(torch.tensor(X).float()), -1)
        return y
    
    def save(self, path):
        torch.save(self.model, path)
        print('保存到:', path)


    def load(self, path):
        self.model = torch.load(path)
        print('加载模型:', path)