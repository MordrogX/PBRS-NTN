import torch
import pandas as pd
import pytorch_lightning as pl
import torch.nn as nn
import tensorflow as tf
import numpy as np
import torch.nn.functional as F
from torchntn import TenorNetworkModule

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tensor = TenorNetworkModule(output_dim=32, input_dim=27)
        self.l1 = nn.Linear(in_features= 32,out_features= 1)

    def forward(self, x):
        scores = self.l1(self.tensor(x[0], x[1]))
        return scores

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self([x[0][0],x[1][0]])
        loss = F.mse_loss(logits[0], y[0])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self([x[0][0],x[1][0]])
        loss = F.mse_loss(logits[0], y[0])
        acc = 1 - min(1, abs(logits[0] - y[0]))
        self.log("val_loss", loss, on_epoch= True)
        self.log("accuracy", acc, on_epoch= True)
        return {"loss": loss, "accuracy": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.003)