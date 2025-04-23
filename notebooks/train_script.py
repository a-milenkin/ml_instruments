#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from lightning import Trainer, LightningModule, LightningDataModule
from clearml import Task

# Initialize ClearML task and configuration
task = Task.init(
    project_name="Image Example", task_name="CIFAR10 Lightning with DataModule"
)


config = {
    "num_epochs": 2,
    "batch_size": 4,
    "dropout": 0.25,
    "learning_rate": 0.001,
    "num_workers": 2,
    "data_dir": "./data",
}
config = task.connect(config)
#print(config)


# Define the LightningDataModule for CIFAR10
class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        # Download data if not already available
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Setup datasets for train, validation, and test
        if stage == "fit" or stage is None:
            self.trainset = datasets.CIFAR10(
                root=self.data_dir, train=True, transform=self.transform
            )
            self.valset = datasets.CIFAR10(
                root=self.data_dir, train=False, transform=self.transform
            )
        if stage == "test" or stage is None:
            self.testset = datasets.CIFAR10(
                root=self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

# Define the LightningModule for the CIFAR10 classifier
class LitCIFAR10Classifier(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.test_accs = []
        # Define the network architecture
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.dropout = nn.Dropout(p=self.hparams.get("dropout", 0.25))
        self.fc3 = nn.Linear(84, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(self.dropout(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        task.get_logger().report_scalar(
            title="val_acc",
            series="val_acc",
            value=acc.item(),
            iteration=self.current_epoch,
        )
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.test_accs.append(acc)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"test_loss": loss, "test_acc": acc}

    def on_test_epoch_end(self):
        # Compute average test accuracy
        avg_acc = sum(self.test_accs) / len(self.test_accs)
        self.log("test_accuracy", avg_acc)

        # Log accuracy in ClearML
        task.get_logger().report_single_value(name="ACC", value=avg_acc.item())
        task.get_logger().report_scalar(
            "Test Accuracy", "Final", avg_acc.item(), iteration=0
        )
        print(f"Final Test Accuracy: {avg_acc.item():.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams.get("learning_rate", 0.001), momentum=0.9
        )
        return optimizer


# Instantiate the DataModule and LightningModule
data_module = CIFAR10DataModule(
    data_dir=config.get("data_dir", "./data"),
    batch_size=config.get("batch_size", 4),
    num_workers=config.get("num_workers", 2),
)
model = LitCIFAR10Classifier(config)


# Create the Trainer using the new Lightning import and run training
trainer = Trainer(max_epochs=config.get("num_epochs", 3))
trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)


# Save the trained model weights
PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)


print("Task ID number is: {}".format(task.id))


task.close()
