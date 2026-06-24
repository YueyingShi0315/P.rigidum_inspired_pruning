#!/usr/bin/env python3
"""
P. rigidum-inspired Structured Channel Pruning for CNNs
========================================================

A PyTorch implementation of structured channel pruning inspired by the
information flow patterns of Physarum rigidum. The core idea is to evaluate
the importance of each convolutional channel by measuring the cosine
dissimilarity between its input and output feature maps. Channels that
preserve the direction of the activation distribution (i.e., have small
cosine distance) are considered essential and are retained, while others
are pruned.

This script provides a self-contained demo on MNIST:
    1. Train a baseline CNN
    2. Prune a fraction of channels using the P. rigidum-inspired metric
    3. Fine-tune the pruned model
    4. Report accuracy before and after pruning

Reference:
    "Physarum rigidum-inspired structured channel pruning for convolutional
     neural networks" (your paper / repository, 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy

# -------------------------------
# Reproducibility & Device
# -------------------------------
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Hyperparameters
# -------------------------------
BATCH_SIZE = 64
EPOCHS_BASELINE = 5
EPOCHS_FINETUNE = 5
LEARNING_RATE = 0.001
PRUNE_RATIO = 0.3                 # fraction of channels to remove (0.0 ~ 1.0)

# -------------------------------
# Data: MNIST
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# -------------------------------
# CNN Model with BatchNorm
# -------------------------------
class CNN(nn.Module):
    def __init__(self, in_channels=1, conv_channels=[32, 64, 128, 256],
                 fc_units=512, num_classes=10):
        super().__init__()
        c1, c2, c3, c4 = conv_channels
        self.conv1 = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(c2)
        self.pool  = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(c3)
        self.conv4 = nn.Conv2d(c3, c4, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(c4)
        self.gap   = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1   = nn.Linear(c4 * 7 * 7, fc_units)
        self.drop  = nn.Dropout(0.5)
        self.fc2   = nn.Linear(fc_units, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

# -------------------------------
# Training & Evaluation Utilities
# -------------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = output.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    return 100. * correct / total

# -------------------------------
# P. rigidum-inspired Pruning Logic
# -------------------------------
def p_rigidum_prune_indices(model, dataloader, prune_ratio):
    """
    Determine which channels to keep in each convolutional layer
    using the P. rigidum-inspired cosine dissimilarity measure.

    For each Conv2d layer we:
      1. Capture input and output feature maps from one mini-batch.
      2. Compute the spatial mean of the whole input, and the spatial
         mean of each output channel.
      3. Measure the cosine similarity between those two scalars
         (broadcast to 1D).  Low dissimilarity (1 - cos_sim) indicates
         the channel preserves the direction of the activation flow,
         mimicking the efficient transport networks of P. rigidum.
      4. Keep the (1 - prune_ratio) fraction of channels with the
         smallest dissimilarity.

    Returns:
        A list of numpy arrays, one per Conv2d layer, containing the
        sorted indices of channels to retain.
    """
    inputs = {}
    outputs = {}

    # Hooks to capture feature maps
    def hook_factory(name, io_type):
        def hook(module, inp, out):
            if io_type == 'in':
                inputs[name] = inp[0].detach()
            else:
                outputs[name] = out.detach()
        return hook

    handles = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(hook_factory(name, 'in')))
            handles.append(m.register_forward_hook(hook_factory(name, 'out')))

    # One forward pass to collect features
    model.eval()
    with torch.no_grad():
        for data, _ in dataloader:
            _ = model(data.to(device))
            break
    for h in handles:
        h.remove()

    keep_masks = []
    for conv_name in inputs.keys():
        in_feat  = inputs[conv_name]
        out_feat = outputs[conv_name]
        C_out = out_feat.shape[1]

        diss = []
        in_mean = in_feat.mean().item()
        for c in range(C_out):
            out_mean = out_feat[:, c, :, :].mean().item()
            in_vec  = torch.tensor([in_mean], device=device)
            out_vec = torch.tensor([out_mean], device=device)
            cos_sim = F.cosine_similarity(in_vec.unsqueeze(0), out_vec.unsqueeze(0)).item()
            diss.append(1.0 - cos_sim)

        diss = np.array(diss)
        n_keep = max(1, int(C_out * (1.0 - prune_ratio)))
        keep_idx = np.sort(np.argsort(diss)[:n_keep])
        keep_masks.append(keep_idx)

    return keep_masks


def apply_pruning(original_model, keep_masks):
    """
    Create a new model with pruned channels based on keep_masks.
    This function hard-codes the architecture; adapt for your own network.
    """
    new_channels = [len(keep_masks[0]), len(keep_masks[1]),
                    len(keep_masks[2]), len(keep_masks[3])]
    pruned = CNN(in_channels=1, conv_channels=new_channels, fc_units=512, num_classes=10).to(device)

    # ---- conv1 ----
    pruned.conv1.weight.data = original_model.conv1.weight.data[keep_masks[0], :, :, :].clone()
    pruned.conv1.bias.data   = original_model.conv1.bias.data[keep_masks[0]].clone()
    pruned.bn1.weight.data   = original_model.bn1.weight.data[keep_masks[0]].clone()
    pruned.bn1.bias.data     = original_model.bn1.bias.data[keep_masks[0]].clone()
    pruned.bn1.running_mean  = original_model.bn1.running_mean[keep_masks[0]].clone()
    pruned.bn1.running_var   = original_model.bn1.running_var[keep_masks[0]].clone()

    # ---- conv2 ----
    pruned.conv2.weight.data = original_model.conv2.weight.data[keep_masks[1], :, :, :][:, keep_masks[0], :, :].clone()
    pruned.conv2.bias.data   = original_model.conv2.bias.data[keep_masks[1]].clone()
    pruned.bn2.weight.data   = original_model.bn2.weight.data[keep_masks[1]].clone()
    pruned.bn2.bias.data     = original_model.bn2.bias.data[keep_masks[1]].clone()
    pruned.bn2.running_mean  = original_model.bn2.running_mean[keep_masks[1]].clone()
    pruned.bn2.running_var   = original_model.bn2.running_var[keep_masks[1]].clone()

    # ---- conv3 ----
    pruned.conv3.weight.data = original_model.conv3.weight.data[keep_masks[2], :, :, :][:, keep_masks[1], :, :].clone()
    pruned.conv3.bias.data   = original_model.conv3.bias.data[keep_masks[2]].clone()
    pruned.bn3.weight.data   = original_model.bn3.weight.data[keep_masks[2]].clone()
    pruned.bn3.bias.data     = original_model.bn3.bias.data[keep_masks[2]].clone()
    pruned.bn3.running_mean  = original_model.bn3.running_mean[keep_masks[2]].clone()
    pruned.bn3.running_var   = original_model.bn3.running_var[keep_masks[2]].clone()

    # ---- conv4 ----
    pruned.conv4.weight.data = original_model.conv4.weight.data[keep_masks[3], :, :, :][:, keep_masks[2], :, :].clone()
    pruned.conv4.bias.data   = original_model.conv4.bias.data[keep_masks[3]].clone()
    pruned.bn4.weight.data   = original_model.bn4.weight.data[keep_masks[3]].clone()
    pruned.bn4.bias.data     = original_model.bn4.bias.data[keep_masks[3]].clone()
    pruned.bn4.running_mean  = original_model.bn4.running_mean[keep_masks[3]].clone()
    pruned.bn4.running_var   = original_model.bn4.running_var[keep_masks[3]].clone()

    # ---- fc1 (re‑weight columns corresponding to kept channels) ----
    keep_cols = []
    for c in keep_masks[3]:
        start = c * 7 * 7
        keep_cols.extend(range(start, start + 7*7))
    pruned.fc1.weight.data = original_model.fc1.weight.data[:, keep_cols].clone()
    pruned.fc1.bias.data   = original_model.fc1.bias.data.clone()
    pruned.fc2.weight.data = original_model.fc2.weight.data.clone()
    pruned.fc2.bias.data   = original_model.fc2.bias.data.clone()

    return pruned

# -------------------------------
# Main Experiment
# -------------------------------
if __name__ == "__main__":
    print("=== 1. Training baseline CNN on MNIST ===")
    baseline = CNN().to(device)
    opt = optim.Adam(baseline.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS_BASELINE + 1):
        train_epoch(baseline, train_loader, opt, criterion)
        acc = evaluate(baseline, test_loader)
        print(f"  Epoch {epoch}/{EPOCHS_BASELINE}  Test Acc: {acc:.2f}%")

    baseline_acc = evaluate(baseline, test_loader)
    print(f"Baseline test accuracy: {baseline_acc:.2f}%\n")

    print(f"=== 2. P. rigidum-inspired pruning ({PRUNE_RATIO*100:.0f}% channels) ===")
    keep_masks = p_rigidum_prune_indices(baseline, test_loader, prune_ratio=PRUNE_RATIO)
    for i, mask in enumerate(keep_masks):
        print(f"  Conv{i+1}: keeping {len(mask)} / {32*(2**i)} channels")

    pruned_model = apply_pruning(baseline, keep_masks)

    print("\n=== 3. Fine-tuning the pruned model ===")
    opt_ft = optim.Adam(pruned_model.parameters(), lr=LEARNING_RATE * 0.1)
    for epoch in range(1, EPOCHS_FINETUNE + 1):
        train_epoch(pruned_model, train_loader, opt_ft, criterion)
        acc = evaluate(pruned_model, test_loader)
        print(f"  Fine-tune Epoch {epoch}/{EPOCHS_FINETUNE}  Test Acc: {acc:.2f}%")

    pruned_acc = evaluate(pruned_model, test_loader)
    print("\n=== Final Results ===")
    print(f"Baseline accuracy : {baseline_acc:.2f}%")
    print(f"Pruned  accuracy  : {pruned_acc:.2f}%")
    print(f"Accuracy drop     : {baseline_acc - pruned_acc:.2f}%")