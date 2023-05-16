import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys
import matplotlib.pyplot as plt


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

            torch.nn.Flatten(),

            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(256, n_out_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x



# file read


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


# Create the original dataset and dataloader
# batch_size = 16
# train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# GAN code
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Generator and Discriminator models for GAN
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU()
        )

    def forward(self, z):
        out = self.model(z)
        out = out.view(out.shape[0], 1, 16, 16, 16)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(32, 128, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool3d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)
        return out

# Set random seed for reproducibility
torch.manual_seed(42)

# Define hyperparameters
latent_dim = 100
num_epochs_gan = 50
num_epochs_main = 50
batch_size = 16

# Initialize the Generator and Discriminator models
generator = Generator(latent_dim)
discriminator = Discriminator()

# Define loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# Train the GAN
for epoch in range(num_epochs_gan):
    for i, (real_images, _) in enumerate(dl_train):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Train Discriminator
        discriminator.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)

        # Real images label as 1, fake images label as 0
        real_labels = torch.full((batch_size, 1), 1.0).to(device)
        fake_labels = torch.full((batch_size, 1), 0.0).to(device)

        # Discriminator loss with real images
        real_output = discriminator(real_images)
        real_loss = criterion(real_output, real_labels)

        # Discriminator loss with fake images
        fake_output = discriminator(fake_images.detach())
        fake_loss = criterion(fake_output, fake_labels)

        # Total discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        generator.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)

        # Generator loss
        gen_output = discriminator(fake_images)
        gen_loss = criterion(gen_output, real_labels)

        gen_loss.backward()
        optimizer_G.step()

        # Print training progress
        if (i + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs_gan}], Step [{i + 1}/{len(dl_train)}], "
                f"Discriminator Loss: {d_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}"
            )

# Generate augmented samples using the trained Generator
num_samples = len(ds_train)
augmented_samples = []
with torch.no_grad():
    for i in range(num_samples):
        z = torch.randn(1, latent_dim).to(device)
        fake_sample = generator(z)
        augmented_samples.append(fake_sample.squeeze().cpu().numpy())

# Convert augmented samples to numpy array
augmented_samples = np.array(augmented_samples)
augmented_samples = augmented_samples.reshape(-1, 1, 16, 16, 16)

# Concatenate augmented samples with original data
x_train_augmented = np.concatenate((x_train, augmented_samples), axis=0)
y_train_augmented = np.concatenate((y_train, y_train), axis=0)

# Update the dataloader with augmented data
ds_train_augmented = torch.utils.data.TensorDataset(torch.Tensor(x_train_augmented),
                                                    torch.Tensor(y_train_augmented))
dl_train_augmented = torch.utils.data.DataLoader(ds_train_augmented, batch_size=16)




# train on epoch function
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

LEARNING_RATE = 0.0001

model = VoxelModel()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()
# device = "cuda" if torch.cuda.is_available()==True else "cpu"
device = "cpu"
epochs = 50

# Train the main model using augmented data
for epoch in range(num_epochs_main):
    train_loss = train_on_epoch(model, dl_train_augmented, optimizer, loss_fn, device)
    train_losses.append(train_loss)
    test_loss = test(model, dl_test, loss_fn, device)
    test_losses.append(test_loss)
    print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")


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