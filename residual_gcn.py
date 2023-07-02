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
import wandb
wandb.login()
dataset = HW3Dataset(root='data/hw3/')
data = dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()
def val_accuracy(pred_y, y):
    """Calculate val accuracy."""
    return ((pred_y == y).sum()).item()

def get_train_mask(batch):
  idxs = batch.n_id
  train_mask = []
  for i,org_id in enumerate(idxs):
    if org_id<80000:
      train_mask.append(i)
  return train_mask

def get_val_mask(batch):
  idxs = batch.n_id
  input_idxs = batch.input_id
  val_mask = []
  for i,org_id in enumerate(idxs):
    if org_id-80000 in input_idxs:
      val_mask.append(i)
  return val_mask


def train_gcn(model, data, epochs, lr=0.001):
    data = data.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    for epoch in range(epochs+1):
      total_loss = 0
      acc = 0
      val_loss = 0
      val_acc = 0
      labels = data.y.flatten()
      out = model(data.x, data.edge_index)
      loss = criterion(out[data.train_mask], labels[data.train_mask])
      total_loss += loss
      acc += accuracy(out[data.train_mask].argmax(dim=1), 
                        labels[data.train_mask])
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      
      model.eval()
      with torch.no_grad():
        labels = data.y.flatten()
        out = model(data.x, data.edge_index)
        val_loss += criterion(out[data.val_mask], labels[data.val_mask])
        val_acc += val_accuracy(out[data.val_mask].argmax(dim=1), 
                      labels[data.val_mask])
      model.train() # set the model back to training mode
      wandb.log({
            "train_acc": acc*100,
            "val_acc": val_acc/len(data.val_mask)*100,
            "train_loss": total_loss,
            "val_loss":  val_loss})
      print('-------------------------------')
      print(f'Epoch {epoch} | Train Loss: {total_loss:.3f} '
            f'| Train Acc: {acc*100:>6.2f}% | Val Loss: '
            f'{val_loss:.2f} | Val Acc: '
            f'{val_acc/len(data.val_mask)*100:.2f}%')
      train_losses.append(total_loss)
      train_accs.append(acc*100)
      val_accs.append(val_acc/len(data.val_mask)*100)
      val_losses.append(val_loss)
      # Check if the current model has the best validation accuracy
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          # Save the current best model
          torch.save(model.state_dict(), 'best_model.pth')
    wandb.finish()
    return train_accs,train_losses,val_accs,val_losses

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
        self.res1 = ResidualGC(hidden_channels,hidden_channels, hidden_channels,3, dropout)    
        self.bridge = GCNConv(hidden_channels, hidden_channels)     
        self.res2 = ResidualGC(hidden_channels,hidden_channels, out_channels,3, dropout)
        self.final_conv = GCNConv(out_channels, out_channels)    
        self.dropout = dropout

    def reset_parameters(self):
        self.init_conv.reset_parameters()
        self.res1.reset_parameters()
        self.bridge.reset_parameters()
        self.res2.reset_parameters()
        self.final_conv.reset_parameters()

    def forward(self, x, adj):
        x = self.init_conv(x,adj)
        x = self.res1(x,adj)
        x = self.bridge(x,adj)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.res2(x,adj)
        x = self.final_conv(x,adj)

        return x
        
        

wandb.init(
      # Set the project where this run will be logged
      project="lab3", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"GCN_residual", 
      save_code=True,
      # Track hyperparameters and run metadata
      config={
      "learning_rate": 0.005,
      "architecture": "GCN Residual",
      "epochs": 1000,
      })

# Create GCN
hidden_channels = 256
in_channels = dataset.num_node_features
out_channels = dataset.num_classes
num_layers = 3
dropout = 0.05
gcn = GCN(in_channels, hidden_channels, out_channels, num_layers, dropout).to(device)
print(gcn)
train_accs,train_losses,val_accs,val_losses = train_gcn(gcn, data, 1000, 0.005)
wandb.finish()

