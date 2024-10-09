import torch
import torch.nn as nn
import torch_geometric.nn as pyg

class WindowGNN(nn.Module):
    def __init__(self, num_layers = 4, in_channel=512):
        super().__init__()

        # Hidden channels for our model
        hidden_channels = 512
        # The output channels of our top 250 genes
        out_channels = 250

        # We have feature data from a pretrained ResNet which was trained on images.
        # We want to get gene expression so we create a MLP to facilitate this
        self.pretransform_layer = pyg.Linear(in_channel, hidden_channels, bias=False)
        self.linear_layer = pyg.Linear(hidden_channels, hidden_channels, bias=False)
        self.graph_learning = nn.Sequential()
        for _ in range(num_layers):
            conv = pyg.GATv2Conv(hidden_channels, hidden_channels)
            self.graph_learning.append(conv)
            

        # Final MLP to convert to gene expression
        self.classifier = pyg.Linear(hidden_channels, out_channels)

    def forward(self, nodes, edges):
        x = self.pretransform_layer(nodes)
        x = x.relu()
        for _ in range(0, 3):
            x = self.linear_layer(x)
            x = x.relu()
        x = self.graph_learning(x, edges)
        out = self.classifier(x)
        return x, out
