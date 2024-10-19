import lightning as L
from train import WinGNNTrain
from dataset import TxGBreastDataset
from torch.utils.data import DataLoader

# Setup data
print("Loading dataset...")
dataset = TxGBreastDataset("../data", [0], 224)
train_loader = DataLoader(dataset, batch_size = 1, shuffle = True)

# Training
print("Training...")
trainer = L.Trainer(max_epochs=100, accelerator="auto")
trainer.fit(model=WinGNNTrain(), train_dataloaders=train_loader)

