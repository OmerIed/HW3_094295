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


def train_gat(model, data, epochs, lr=0.001, accumulation_steps=1, batch_size=32):
    data = data.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0

    data.batch_size = batch_size  # Decrease the batch size

    for epoch in range(epochs + 1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        # Create data loader with smaller batch size
        loader = torch_geometric.loader.DataLoader(data, batch_size=batch_size, shuffle=True)

        for batch in loader:
            batch = batch.to(device)

            labels = batch.y.flatten()
            out = model(batch.x.half(), batch.edge_index)
            loss = criterion(out[batch.train_mask], labels[batch.train_mask])
            total_loss += loss
            acc += accuracy(out[batch.train_mask].argmax(dim=1), labels[batch.train_mask])

            loss /= accumulation_steps  # Scale the loss by the number of accumulation steps
            loss.backward()

            if (epoch + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Clear GPU memory
            torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            labels = data.y.flatten()
            out = model(data.x.half(), data.edge_index)
            val_loss += criterion(out[data.val_mask], labels[data.val_mask])
            val_acc += val_accuracy(out[data.val_mask].argmax(dim=1), labels[data.val_mask])

        model.train()
        wandb.log({
            "train_acc": acc * 100 / len(data.train_mask),
            "val_acc": val_acc * 100 / len(data.val_mask),
            "train_loss": total_loss,
            "val_loss": val_loss})

        print('-------------------------------')
        print(f'Epoch {epoch} | Train Loss: {total_loss:.3f} | Train Acc: {acc * 100 / len(data.train_mask):>6.2f}% | '
              f'Val Loss: {val_loss:.2f} | Val Acc: {val_acc * 100 / len(data.val_mask):.2f}%')

        train_losses.append(total_loss)
        train_accs.append(acc * 100 / len(data.train_mask))


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 128
        self.in_head = 16
        self.out_head = 1
        
        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.2)
        self.conv2 = GATConv(self.hid*self.in_head, dataset.num_classes, concat=False,
                             heads=self.out_head, dropout=0.2)
        self.fc1 = torch.nn.Linear(1024,256)
        self.fc2 = torch.nn.Linear(256,40)

    def forward(self,x, edge_index):
        
        # Dropout before the GAT layer is used to avoid overfitting
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # x = self.conv2(x, edge_index)
        #x = self.fc1(x)
        #x = self.fc2(x)
        return x


wandb.init(
      # Set the project where this run will be logged
      project="lab3", 
      name="GAT_2layers_single_batch",
      save_code=True,
      # Track hyperparameters and run metadata
      config={
      "learning_rate": 0.01,
      "architecture": "GAT",
      "epochs": 1000,
      })

# Create GCN
gat = GAT().half()
gat = gat.to(device)
print(gat)
train_accs,train_losses,val_accs,val_losses = train_gat(gat, data, 1000, 0.01)
wandb.finish()

