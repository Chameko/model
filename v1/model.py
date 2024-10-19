import gnn
import torch
import torch.nn as nn
import torch_geometric.nn as pyg
import numpy as np
from torchvision import models

class WinGNN(nn.Module):
    def __init__(self, num_layers = 4):
        super().__init__()
        # Encode the data into feature space. We use a resnet with the classifying layer removed
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        # Graph learning
        self.gnn = WinGNN(num_layers)

    def forward(self, windows):
        embedded = torch.Tensor()
        window_pos = torch.Tensor()
        for window in windows:
            window_embed = self.encoder(window["img"])
            embedded = torch.cat(window_embed)
            window_pos = torch.cat(window["pos"])
        edges = pyg.radius_graph(window_pos, np.sqrt(2), None, False, max_num_neighbors=5, flow="source_to_target", num_workers=1)
        out = self.gnn(embedded, edges)
        return out
