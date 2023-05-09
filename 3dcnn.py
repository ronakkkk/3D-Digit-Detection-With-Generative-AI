import sys
import os
import h5py
from collections import Counter
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm import tqdm_notebook

import torch

# 3D model
class VoxelModel(torch.nn.Module):
    def __init__(self, n_out_classes=10):
        super(VoxelModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.MaxPool3d(kernel_size=2),

            torch.nn.Conv3d(32, 128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.MaxPool3d(kernel_size=2),

            #             torch.nn.Conv3d(32,64, kernel_size=3),
            #             torch.nn.ReLU(),
            #             torch.nn.Dropout(0.3),

            #             torch.nn.Conv3d(64,128, kernel_size=3),
            #             torch.nn.ReLU(),
            #             torch.nn.Dropout(0.3),

            #             torch.nn.Conv3d(128,256, kernel_size=3),
            #             torch.nn.ReLU(),
            #             torch.nn.Dropout(0.3),
            #             torch.nn.MaxPool3d(kernel_size=2),

            #             torch.nn.Conv3d(256,512, kernel_size=3),
            #             torch.nn.ReLU(),

            torch.nn.Flatten(),

            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(256, n_out_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


mdl = VoxelModel()
x = torch.randn(2, 1, 16, 16, 16)
# mdl(x).shape

f = h5py.File("full_dataset_vectors.h5/full_dataset_vectors.h5","r")
x_train = f["X_train"]
y_train = f["y_train"]

x_test = f["X_test"]
y_test = f["y_test"]

# x_train.shape, y_train.shape, x_test.shape, y_test.shape

x_train = np.array(x_train).reshape(-1, 1, 16,16,16) # Extra dimension of 1 is added to indicate channels.
y_train = np.array(y_train)

x_test = np.array(x_test).reshape(-1, 1, 16,16,16) # Extra dimension of 1 is added to indicate channels.
y_test = np.array(y_test)


ds_train = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=16)

ds_test = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=16)

LEARNING_RATE = 0.0001

model = VoxelModel()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()
device = "cuda" if torch.cuda.is_available()==True else "cpu"
epochs = 30

print(f"Using device {device}")

def train_on_epoch(model, dl_train, optimizer, loss_fn, device):
    model.train()
    model = model.to(device)
    losses = []
    N = len(dl_train)
    for i, (x, y) in enumerate(dl_train):
        x, y = x.to(device).float(), y.to(device).float()

        y_pred = model(x)
        loss = loss_fn(y_pred, y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        sys.stdout.write(f"\r{i}/{N} train_loss:{loss.cpu().detach().numpy()}")
        sys.stdout.flush()
    mean_loss = np.mean(losses)
    return mean_loss

def test(model, dl_test, loss_fn, device):
    model.eval()
    model = model.to(device)
    losses = []
    N = len(dl_test)
    for i, (x, y) in enumerate(dl_test):
        x, y = x.to(device).float(), y.to(device).float()

        y_pred = model(x)
        loss = loss_fn(y_pred, y.long())

        losses.append(loss.detach().cpu().numpy())
        sys.stdout.write(f"\r{i}/{N} train_loss:{loss.cpu().detach().numpy()}")
        sys.stdout.flush()
    mean_loss = np.mean(losses)
    return mean_loss

train_losses = []
test_losses = []
for epoch in range(epochs):
    train_loss = train_on_epoch(model, dl_train, optimizer, loss_fn, device)
    train_losses.append(train_loss)
    #     print(f"\rTrain Epoch:{epoch+1} loss:{train_loss}")

    test_loss = test(model, dl_test, loss_fn, device)
    test_losses.append(test_loss)
    print(f"\rEpoch:{epoch + 1} train_loss:{train_loss} test_loss:{test_loss}")


plt.plot(train_losses, label="train_losses")
plt.plot(test_losses, label="test_losses")
plt.legend(loc="upper right")
plt.show()

model.eval()
_x_test = torch.Tensor(x_test)
_x_test = _x_test.to(device)
y_test_pred = model(_x_test)
y_test_pred = torch.argmax(y_test_pred, axis=-1)
y_test_pred = y_test_pred.detach().cpu().numpy()

from sklearn.metrics import classification_report

print(classification_report(y_test, y_test_pred))