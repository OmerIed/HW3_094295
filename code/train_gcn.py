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
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx
from dataset import HW3Dataset
import wandb
wandb.login()
dataset = HW3Dataset(root='data/hw3/')
data = dataset[0]

batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = NeighborLoader(
    data,
    num_neighbors=[20,50],
    batch_size=batch_size,
    input_nodes=data.train_mask,
    shuffle=True
)
val_loader = NeighborLoader(
    data,
    num_neighbors=[20,50],
    batch_size=batch_size,
    input_nodes=data.val_mask,
    shuffle=True
)

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
    train_loader_length = len(train_loader)
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

      # Train on batches
      for batch_idx, batch in enumerate(train_loader):
        # train_mask = get_train_mask(batch)
        train_mask = [x for x in range(batch.batch_size)]
        labels = batch.y.flatten()
        labels_onehot = F.one_hot(labels, num_classes=40).float()
        out = model(batch.x, batch.edge_index)
        # print(batch.train_mask, )
        # print(out[train_mask], out[train_mask].shape)
        # print(labels_onehot[train_mask], labels_onehot[train_mask].shape)
        # print(labels[train_mask])
        # print(out[train_mask])
        loss = criterion(out[train_mask], labels[train_mask])
        total_loss += loss
        acc += accuracy(out[train_mask].argmax(dim=1), 
                        labels[train_mask])
        # if batch_idx % 900 == 0:
        #   print('predictions and gt',out[train_mask].argmax(dim=1), 
        #                 labels[train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # # Validation
        # if val_mask:
        #   val_loss += criterion(out[val_mask], labels_onehot[val_mask])
        #   val_acc += val_accuracy(out[val_mask].argmax(dim=1), 
        #                       labels[val_mask])
        
        # print(f'for batch {batch_idx}|{train_loader_length} acc is: {val_acc}')
      # Get validation metrics
      model.eval()
      with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
          # val_mask = get_val_mask(batch)
          val_mask = [x for x in range(batch.batch_size)]
          labels = batch.y.flatten()
          labels_onehot = F.one_hot(labels, num_classes=40).float()
          out = model(batch.x, batch.edge_index)
          val_loss += criterion(out[val_mask], labels[val_mask])
          val_acc += val_accuracy(out[val_mask].argmax(dim=1), 
                          labels[val_mask])
      model.train() # set the model back to training mode
      wandb.log({
            "train_acc": acc/train_loader_length*100,
            "val_acc": val_acc/len(data.val_mask)*100,
            "train_loss": total_loss/train_loader_length,
            "val_loss":  val_loss/len(val_loader)})
      print('-------------------------------')
      print(f'Epoch {epoch} | Train Loss: {total_loss/train_loader_length:.3f} '
            f'| Train Acc: {acc/train_loader_length*100:>6.2f}% | Val Loss: '
            f'{val_loss/len(val_loader):.2f} | Val Acc: '
            f'{val_acc/len(data.val_mask)*100:.2f}%')
      train_losses.append(total_loss/train_loader_length)
      train_accs.append(acc/train_loader_length*100)
      val_accs.append(val_acc/len(data.val_mask)*100)
      val_losses.append(val_loss/len(val_loader))
      # Check if the current model has the best validation accuracy
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          # Save the current best model
          torch.save(model.state_dict(), 'best_model.pth')
    wandb.finish()
    return train_accs,train_losses,val_accs,val_losses


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 256, improved=True)
        self.conv2 = GCNConv(256,128, improved=True)
        self.conv3 = GCNConv(128,64, improved=True)
        self.conv4 = GCNConv(64, dataset.num_classes, improved=True)
    def forward(self, x, edge_index):
        # x: Node feature matrix 
        # edge_index: Graph connectivity matrix 
        #x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        return x


wandb.init(
      # Set the project where this run will be logged
      project="lab3", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"GCN_02_4layers_train_only_input_improved_all", 
      save_code=True,
      # Track hyperparameters and run metadata
      config={
      "learning_rate": 0.001,
      "architecture": "GCN",
      "epochs": 100,
      })

# Create GCN
gcn = GCN().to(device)
print(gcn)
train_accs,train_losses,val_accs,val_losses = train_gcn(gcn, data, 100, 0.001)
wandb.finish()
