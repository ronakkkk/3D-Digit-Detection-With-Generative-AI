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

f = h5py.File(f"full_dataset_vectors.h5/full_dataset_vectors.h5","r")
x_train = f["X_train"]
y_train = f["y_train"]

x_test = f["X_test"]
y_test = f["y_test"]

def vector_to_voxel(vector, shape=(16,16,16)):
    vec_shape = list(vector.shape[:-1]) + list(shape)
    voxel = vector.reshape(*vec_shape)
    return voxel

def voxel_to_pointcloud(voxel, num_points=None, shuffle=True):
    assert len(voxel.shape)==3, f"Voxel should be a 3D tensor. Given shape {voxel.shape}!=3"
    x, y, z = np.nonzero(voxel)
    point_cloud = np.concatenate([np.expand_dims(x,axis=1), np.expand_dims(y,axis=1), np.expand_dims(z,axis=1)], axis=1)
    if num_points is not None:
        if point_cloud.shape[0] >= num_points:
            ids = list(range(point_cloud.shape[0]))
            ids = ids[:num_points] if shuffle==False else np.random.permutation(ids)[:num_points]
            point_cloud = point_cloud[ids,:]
        else:
            diff = num_points - len(point_cloud)
            padding = np.zeros([diff, 3])
            point_cloud = np.concatenate([point_cloud, padding], axis=0)
            ids = list(range(len(point_cloud)))
            ids = ids[:num_points] if shuffle==False else np.random.permutation(ids)[:num_points]
            point_cloud = point_cloud[ids,:]
    return point_cloud


def vector_to_pointcloud(vector, num_points=None, shuffle=True):
    voxel = vector_to_voxel(vector)
    voxel = np.squeeze(voxel)
    point_cloud = voxel_to_pointcloud(voxel, num_points, shuffle)
    return point_cloud


def plot_3d_digit(point_cloud, digit, size=1, opacity=0.3):
    if isinstance(point_cloud, np.ndarray):
        df = pd.DataFrame(point_cloud, columns=["x", "y", "z"])
    fig = px.scatter_3d(df, x="x", y="y", z="z",
                        # size=[size]*len(df),
                        opacity=opacity,
                        title=f"Current digit is {digit}")
    fig.update_traces(marker_size=size)
    fig.show()
idx = np.random.randint(0, len(x_train), 1)
print(f"Reading {idx} sample from x_train")

vec = x_train[idx]
digit_label = y_train[idx]
pc = vector_to_pointcloud(vec)

plot_3d_digit(pc, digit_label, size=3)

point_count = dict([(i,[]) for i in range(10)])
for vec, digit in tqdm_notebook(zip(x_train, y_train)):
    digit = int(digit)
    pc = vector_to_pointcloud(vec)
    # point_count.append([len(pc), digit])
    lst = point_count.get(digit,[])
    lst.append(len(pc))
    point_count[digit] = lst


plt.figure(figsize=(10,13))
for digit, lst in tqdm_notebook(point_count.items()):
    plt.subplot(4,3, digit+1)
    sns.histplot(lst, bins=10)
    plt.title(f"Digit {digit}")
plt.show()

NUM_POINTS = 1000

temp = []
for vec in tqdm_notebook(x_train):
    pc = vector_to_pointcloud(vec, num_points=NUM_POINTS, shuffle=True)
    temp.append(pc)
x_train = np.array(temp)
print(f"x_train after point cloud conversion {x_train.shape}")

temp = []
for vec in tqdm_notebook(x_test):
    pc = vector_to_pointcloud(vec, num_points=NUM_POINTS, shuffle=True)
    temp.append(pc)
x_test = np.array(temp)
print(f"x_test after point cloud conversion {x_test.shape}")


class MLPModel(torch.nn.Module):
    def __init__(self, n_out_classes=10, num_points=1000):
        super(MLPModel, self).__init__()
        self.dense1 = torch.nn.Linear(3, 32)
        self.dense2 = torch.nn.Linear(32, 64)
        self.dense3 = torch.nn.Linear(64, 128)
        self.dense4 = torch.nn.Linear(128, 256)

        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.bn4 = torch.nn.BatchNorm1d(256)

        self.act = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(kernel_size=num_points)

        self.fc1 = torch.nn.Linear(256, 512)
        self.fc2 = torch.nn.Linear(512, n_out_classes)

    def forward(self, x):
        x = self.dense1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)

        x = self.dense2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)

        x = self.dense3(x)
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)

        x = self.dense4(x)
        x = self.bn4(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)

        x = self.maxpool(x.transpose(1, 2))

        x = torch.squeeze(x)

        x = self.act(self.fc1(x))
        x = self.fc2(x)

        return x


mdl = MLPModel()
x = torch.randn(2, 1000, 3)
mdl(x).shape


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

ds_train = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=16)

ds_test = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=16)

LEARNING_RATE = 0.0001

model = MLPModel()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()
device = "cpu"
epochs = 30

print(f"Using device {device}")

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