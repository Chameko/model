import os
from model import WinGNN
from torch import optim, nn, utils, Tensor
import lightning as L

class WinGNNTrain(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = WinGNN(4)
    
    def training_step(self, batch, batch_idx):
        out = self.model(batch["img"])
        # MSE loss
        loss = nn.functional.mse_loss(out, batch["counts"])
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.adam.Adam(self.parameters(), lr=1e-3)
        return optimizer