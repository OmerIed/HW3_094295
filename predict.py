import numpy as np
import pandas as pd
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric
from torch.nn import Parameter
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import urllib.request
import tarfile
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_networkx
from torch_geometric.nn import DenseSAGEConv
from dataset import HW3Dataset

dataset = HW3Dataset(root='data/hw3/')
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


normalize = True
class ResidualGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, skip_n_channels,dropout):
        super(ResidualGC, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(skip_n_channels):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.compress = SAGEConv(hidden_channels + in_channels, hidden_channels)
        
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.compress.reset_parameters()
            
    def forward(self, x, adj):
        x_i = x #identity
        for conv in self.convs[:-1]:
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.cat([x_i,x],1)
        
        x = self.compress(x, adj)
        x = self.convs[-1](x, adj)
        return x
        
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()
        self.init_conv = GCNConv(in_channels, hidden_channels)
        self.res1 = ResidualGC(hidden_channels,hidden_channels, hidden_channels,num_layers, dropout)    
        self.bridge = GCNConv(hidden_channels, hidden_channels)     
        self.res2 = ResidualGC(hidden_channels,hidden_channels, out_channels,num_layers, dropout)
        #self.bridge2 = GCNConv(hidden_channels/2, hidden_channels/2)     
        #self.res3 = ResidualGC(hidden_channels/2,hidden_channels/2, out_channels,3, dropout)
        self.final_conv = GCNConv(out_channels, out_channels)    
        self.dropout = dropout

    def reset_parameters(self):
        self.init_conv.reset_parameters()
        self.res1.reset_parameters()
        self.bridge.reset_parameters()
        self.res2.reset_parameters()
        #self.final_conv.reset_parameters()
        #self.res3.reset_parameters()
        self.bridge2.reset_parameters()

    def forward(self, x, adj):
        x = self.init_conv(x,adj)
        x = self.res1(x,adj)
        x = self.bridge(x,adj)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.res2(x,adj)
        #x = self.bridge2(x,adj)
        #x = self.res3(x,adj)
        x = self.final_conv(x,adj)

        return x

def predict(model, data):
    data = data.to(device)
    print(data)
    with torch.no_grad():
        output = model(data.x, data.edge_index)
    predictions = output.argmax(dim=1)
    # Create a DataFrame with the predictions
    df = pd.DataFrame({'idx': np.arange(0,len(data.x)), 'prediction': predictions.cpu().numpy()})

    # Save the DataFrame to a CSV file
    df.to_csv('predictions.csv', index=False)

hidden_channels = 256
in_channels = dataset.num_node_features
out_channels = dataset.num_classes
num_layers = 3
dropout = 0.05
gcn = GCN(in_channels, hidden_channels, out_channels, num_layers, dropout).to(device)
gcn.load_state_dict(torch.load('best_model.pth'))
gcn.eval()
predict(gcn, data)