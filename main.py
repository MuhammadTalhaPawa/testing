#!/usr/bin/env python
# coding: utf-8

# # Local Laptop Version: Multi-Size, Multi-Run, Auto-Resume Training Notebook
# 
# This notebook has been converted from the Kaggle version to run locally from your laptop terminal or Jupyter.
# 
# Current setup:
# 
# 1. Model selected by default: **ResNet-34**.
# 2. Dataset paths are controlled by editable variables in Section 3.
# 3. Results are saved under a local output folder by default.
# 4. Training automatically uses CUDA GPU when available, then Apple MPS when available, otherwise CPU.
# 5. DataLoader settings are configurable for Windows/local CPU stability.
# 6. ZIP backups are created in the local output folder.
# 

# In[21]:


# ============================================================
# 1. Local Environment Check
# ============================================================

import os
import argparse
from pathlib import Path

NOTEBOOK_DIR = Path.cwd()


# In[22]:


# ============================================================
# 1. Import Libraries
# ============================================================

import os
import json
import time
import copy
import random
import shutil
import zipfile
import warnings
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

try:
    from tqdm.auto import tqdm as tqdm_progress
except ImportError:
    class tqdm_progress:
        """Small fallback progress bar used when tqdm is not installed."""
        def __init__(self, iterable=None, total=None, desc=None, unit="it", leave=True, **kwargs):
            self.iterable = iterable
            self.total = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
            self.desc = desc or ""
            self.unit = unit
            self.leave = leave
            self.current = 0
            self.postfix = ""
            self.start_time = time.time()

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
            self.close()

        def __enter__(self):
            self._render()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        def update(self, n=1):
            self.current += n
            self._render()

        def set_postfix(self, values=None, **kwargs):
            data = values or kwargs
            self.postfix = " | " + ", ".join(f"{key}: {value}" for key, value in data.items()) if data else ""
            self._render()

        def close(self):
            self._render(final=True)
            if self.leave:
                sys.stderr.write("\n")
            else:
                sys.stderr.write("\r" + " " * 100 + "\r")
            sys.stderr.flush()

        def _render(self, final=False):
            elapsed = max(time.time() - self.start_time, 0.001)
            if self.total:
                ratio = min(self.current / self.total, 1.0)
                filled = int(24 * ratio)
                bar = "#" * filled + "-" * (24 - filled)
                text = f"\r{self.desc}: |{bar}| {self.current}/{self.total} {self.unit} [{elapsed:.0f}s]{self.postfix}"
            else:
                text = f"\r{self.desc}: {self.current} {self.unit} [{elapsed:.0f}s]{self.postfix}"
            sys.stderr.write(text)
            sys.stderr.flush()

warnings.filterwarnings("ignore")


# In[23]:


# ============================================================
# 3. Main Configuration for Local Laptop / Terminal Runs
# ============================================================

# Dataset paths are auto-detected for the current local folder layout:
#   dataset/
#     The Wildfire Dataset/The Wildfire Dataset 32x32|64x64|128x128
#     Forest Fire Dataset/Forest Fire Dataset 32x32|64x64|128x128
# Leave DATASET_ROOT blank unless you move the dataset elsewhere.

BASE_DIR = Path.cwd()

# Main editable paths.
DATASET_ROOT = r""  # Optional override. Example: r"D:\Datasets\forest-fire-datasets-32-64-128"
PROJECT_ROOT = str(BASE_DIR / "local_training_outputs")

# Optional: set these directly if your folder names differ from the current dataset.
MAIN_DATASET_ROOT = r""      # Example: r"D:\Datasets\dataset\The Wildfire Dataset"
EXTERNAL_DATASET_ROOT = r""  # Example: r"D:\Datasets\dataset\Forest Fire Dataset"

AUTO_ZIP_CURRENT_MODEL_AFTER_TRAINING = True
CREATE_ZIP_BACKUP_AFTER_EACH_RUN = False

# Local ZIP/output folders.
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "training_results")
BACKUP_ZIP_ROOT = os.path.join(PROJECT_ROOT, "training_result_zips")


def first_existing_path(candidates):
    """Return the first existing path from a list, or an empty string if none exists."""
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return str(candidate)
    return ""


# If DATASET_ROOT is blank, try local locations relative to the notebook folder.
# The current dataset in this workspace uses BASE_DIR / "dataset".
if not DATASET_ROOT:
    DATASET_ROOT = first_existing_path([
        BASE_DIR / "dataset",
        BASE_DIR / "datasets",
        BASE_DIR / "data",
        BASE_DIR / "forest-fire-datasets-32-64-128",
        BASE_DIR / "datasets" / "forest-fire-datasets-32-64-128",
        BASE_DIR / "dataset" / "forest-fire-datasets-32-64-128",
        BASE_DIR / "data" / "forest-fire-datasets-32-64-128",
    ])

if not DATASET_ROOT:
    DATASET_ROOT = str(BASE_DIR / "PATH_TO_DATASET_ROOT")

if not MAIN_DATASET_ROOT:
    MAIN_DATASET_ROOT = first_existing_path([
        os.path.join(DATASET_ROOT, "The Wildfire Dataset"),
        os.path.join(DATASET_ROOT, "the wildfire dataset"),
    ]) or os.path.join(DATASET_ROOT, "The Wildfire Dataset")

if not EXTERNAL_DATASET_ROOT:
    EXTERNAL_DATASET_ROOT = first_existing_path([
        os.path.join(DATASET_ROOT, "Forest Fire Dataset"),
        os.path.join(DATASET_ROOT, "forest fire dataset"),
    ]) or os.path.join(DATASET_ROOT, "Forest Fire Dataset")

# Select model here.
# Available options:
#   "resnet18"
#   "resnet34"
#   "resnet50"
#   "mobilenetv2"
#   "mobilenetv3large"
#   "efficientnetb0"
#   "efficientnetb1"
#   "convnexttiny"
#   "vgg16"
#   "vgg19"
#   "densenet121"
MODEL_NAME = "resnet34"

# Laptop-friendly training defaults. Increase these if your GPU/CPU can handle it.
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 0.001
NUM_RUNS = 5
SEED = 42

# Choose which dataset sizes to run. Use [32], [64], [128], or any combination.
DATASET_SIZES_TO_RUN = [32, 64, 128]

# Resume settings.
RESUME_TRAINING = True
SKIP_COMPLETED_RUNS = True
SAVE_CHECKPOINT_EVERY_EPOCH = True

# Device and local resource settings.
USE_MULTI_GPU = True
USE_MIXED_PRECISION = True  # Used only on CUDA GPUs.

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
CPU_COUNT = os.cpu_count() or 1

# Windows notebooks/nbconvert are usually more reliable with 0 workers.
# You can try 2, 4, or more after the notebook works end-to-end.
NUM_WORKERS = 0 if os.name == "nt" else min(4, CPU_COUNT)
PIN_MEMORY = DEVICE.type == "cuda"
PERSISTENT_WORKERS = NUM_WORKERS > 0

def print_runtime_config():
    """Print the active runtime configuration for a training run."""
    print("Device:", DEVICE)
    print("CPU count:", CPU_COUNT)
    print("GPU count:", GPU_COUNT)
    if GPU_COUNT > 0:
        for gpu_idx in range(GPU_COUNT):
            print(f"GPU {gpu_idx}:", torch.cuda.get_device_name(gpu_idx))
    print("Selected model:", MODEL_NAME)
    print("Batch size:", BATCH_SIZE)
    print("Num workers:", NUM_WORKERS)
    print("Mixed precision:", USE_MIXED_PRECISION and DEVICE.type == "cuda")
    print("Project root:", PROJECT_ROOT)
    print("Dataset root:", DATASET_ROOT)
    print("Main dataset root:", MAIN_DATASET_ROOT)
    print("External dataset root:", EXTERNAL_DATASET_ROOT)
    print("Results root:", RESULTS_ROOT)
    print("Backup zip root:", BACKUP_ZIP_ROOT)


# In[24]:


# ============================================================
# 4. Dataset Configurations for 32x32, 64x64, and 128x128
# ============================================================

# Current dataset layout:
# - Main training/validation/internal-test data comes from The Wildfire Dataset.
# - External test data comes from Forest Fire Dataset.
# - Both datasets contain 32x32, 64x64, and 128x128 folders.
DATASET_CONFIGS = [
    {
        "dataset_name": "wildfire_32x32",
        "image_size": 32,
        "main_dataset_path": os.path.join(MAIN_DATASET_ROOT, "The Wildfire Dataset 32x32"),
        "external_dataset_path": os.path.join(EXTERNAL_DATASET_ROOT, "Forest Fire Dataset 32x32")
    },
    {
        "dataset_name": "wildfire_64x64",
        "image_size": 64,
        "main_dataset_path": os.path.join(MAIN_DATASET_ROOT, "The Wildfire Dataset 64x64"),
        "external_dataset_path": os.path.join(EXTERNAL_DATASET_ROOT, "Forest Fire Dataset 64x64")
    },
    {
        "dataset_name": "wildfire_128x128",
        "image_size": 128,
        "main_dataset_path": os.path.join(MAIN_DATASET_ROOT, "The Wildfire Dataset 128x128"),
        "external_dataset_path": os.path.join(EXTERNAL_DATASET_ROOT, "Forest Fire Dataset 128x128")
    }
]

def print_dataset_configs():
    """Print dataset paths used by the training run."""
    for cfg in DATASET_CONFIGS:
        print("Dataset:", cfg["dataset_name"])
        print("  Main path    :", cfg["main_dataset_path"])
        print("  External path:", cfg["external_dataset_path"])


# In[25]:


# ============================================================
# 5. Helper Functions for Reproducibility, Folders, Resume, and Backups
# ============================================================

def set_seed(seed=42):
    """Set random seeds so experiments are more repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dir(path):
    """Create a folder if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data, path):
    """Save a Python dictionary/list as a JSON file."""
    make_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path, default=None):
    """Load JSON safely. Return default if file does not exist."""
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def torch_load_safe(path, map_location=None):
    """
    Load a PyTorch checkpoint safely across different PyTorch versions.

    Newer PyTorch versions may use weights_only=True by default in some cases.
    Our checkpoints contain extra metadata such as class names, history, and config,
    so we explicitly request weights_only=False when the installed PyTorch supports it.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not have the weights_only argument.
        return torch.load(path, map_location=map_location)


def get_base_model(model):
    """Return the real model when it is wrapped by nn.DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


def get_model_state_dict(model):
    """Save checkpoint weights without DataParallel's module. prefix."""
    return get_base_model(model).state_dict()


def load_model_state_dict(model, state_dict):
    """Load weights saved from either DataParallel or a normal model."""
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    get_base_model(model).load_state_dict(state_dict)


def maybe_wrap_data_parallel(model):
    """Use all visible CUDA GPUs when USE_MULTI_GPU=True."""
    if DEVICE.type == "cuda" and USE_MULTI_GPU and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        print(f"Using nn.DataParallel on GPUs: {device_ids}")
        return nn.DataParallel(model, device_ids=device_ids)

    print("Using single device training.")
    return model


def save_text(text, path):
    """Save text into a .txt file."""
    make_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text))


def append_text(text, path):
    """Append one line of text to a log file."""
    make_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(str(text) + "\n")


def resolve_existing_path(path):
    """
    Try slightly different path spellings.
    This helps if folders use x or Ã— in names.
    """
    candidates = [
        path,
        path.replace("x", "Ã—"),
        path.replace("Ã—", "x"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return path


def find_split_folder(dataset_path, split_names):
    """
    Find a split folder even if its name is slightly different.
    Example: valid, val, validation, training, testing.
    """
    dataset_path = resolve_existing_path(dataset_path)

    for name in split_names:
        candidate = os.path.join(dataset_path, name)
        if os.path.exists(candidate):
            return candidate

    # Case-insensitive search.
    if os.path.exists(dataset_path):
        available = os.listdir(dataset_path)
        lower_map = {item.lower(): item for item in available}

        for name in split_names:
            if name.lower() in lower_map:
                return os.path.join(dataset_path, lower_map[name.lower()])

    return None


def check_dataset_path(dataset_path):
    """Print available folders inside a dataset path for debugging."""
    dataset_path = resolve_existing_path(dataset_path)
    print("Checking:", dataset_path)

    if not os.path.exists(dataset_path):
        print("Path not found.")
        return

    print("Available folders/files:")
    for item in os.listdir(dataset_path):
        print(" -", item)


def save_training_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    epoch,
    history,
    best_valid_acc,
    best_epoch,
    epochs_without_improvement,
    class_names,
    num_classes,
    run_config
):
    """
    Save full training state after an epoch.
    This file allows training to resume if the run is interrupted locally.
    """
    make_dir(os.path.dirname(checkpoint_path))

    torch.save({
        "model_name": run_config["model_name"],
        "dataset_name": run_config["dataset_name"],
        "image_size": run_config["image_size"],
        "run_name": run_config["run_name"],
        "epoch": int(epoch),
        "model_state_dict": get_model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "best_valid_acc": float(best_valid_acc),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "class_names": class_names,
        "num_classes": int(num_classes),
        "run_config": run_config
    }, checkpoint_path)


def load_training_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """
    Load full training state.
    Returns checkpoint dictionary and updates model/optimizer/scheduler.
    """
    checkpoint = torch_load_safe(checkpoint_path, map_location=device)
    load_model_state_dict(model, checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def zip_folder(source_dir, zip_path):
    """Zip one completed run folder into the local backup folder."""
    make_dir(os.path.dirname(zip_path))
    base_name = zip_path.replace(".zip", "")

    if os.path.exists(zip_path):
        os.remove(zip_path)

    shutil.make_archive(base_name=base_name, format="zip", root_dir=source_dir)
    return zip_path


def zip_current_model_results():
    """Create/update the current model ZIP in the local backup folder."""
    model_results_dir = os.path.join(RESULTS_ROOT, MODEL_NAME)

    if not os.path.exists(model_results_dir):
        print("Current model results folder does not exist yet:", model_results_dir)
        return []

    zip_roots = [BACKUP_ZIP_ROOT]

    saved_zip_paths = []
    for zip_root in dict.fromkeys(zip_roots):
        make_dir(zip_root)
        model_zip_path = os.path.join(zip_root, f"{MODEL_NAME}_complete_results.zip")

        if os.path.exists(model_zip_path):
            os.remove(model_zip_path)

        shutil.make_archive(
            base_name=model_zip_path.replace(".zip", ""),
            format="zip",
            root_dir=model_results_dir
        )
        saved_zip_paths.append(model_zip_path)

    print("Current model results zipped successfully.")
    print("Model results folder:", model_results_dir)
    for zip_path in saved_zip_paths:
        print("ZIP backup path:", zip_path)

    return saved_zip_paths


# for folder in make_drive_dirs_later:
#     make_dir(folder)



# In[26]:


# ============================================================
# 5. Dataset Loading Functions
# ============================================================

def get_transforms(image_size):
    """Create train and test transforms for a selected image size."""

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transforms, eval_transforms


def load_main_datasets(dataset_path, image_size):
    """
    Load train, validation, and test folders from the main dataset.
    Supports folder names like train/training, val/valid/validation, test/testing.
    """

    dataset_path = resolve_existing_path(dataset_path)

    train_dir = find_split_folder(dataset_path, ["train", "training", "Train", "Training"])
    valid_dir = find_split_folder(dataset_path, ["valid", "val", "validation", "Valid", "Val", "Validation"])
    test_dir  = find_split_folder(dataset_path, ["test", "testing", "Test", "Testing"])

    if train_dir is None:
        raise FileNotFoundError(f"Training folder not found inside: {dataset_path}")
    if valid_dir is None:
        raise FileNotFoundError(f"Validation folder not found inside: {dataset_path}")
    if test_dir is None:
        raise FileNotFoundError(f"Test folder not found inside: {dataset_path}")

    train_tf, eval_tf = get_transforms(image_size)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_tf)
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=eval_tf)
    test_dataset  = datasets.ImageFolder(root=test_dir,  transform=eval_tf)

    return train_dataset, valid_dataset, test_dataset, train_dir, valid_dir, test_dir


def load_external_test_dataset(external_dataset_path, image_size):
    """
    Load the second dataset for external testing only.
    It first looks for Testing/test folder.
    If no testing folder exists, it uses the dataset path itself.
    """

    external_dataset_path = resolve_existing_path(external_dataset_path)
    _, eval_tf = get_transforms(image_size)

    external_test_dir = find_split_folder(external_dataset_path, ["test", "testing", "Test", "Testing"])

    if external_test_dir is None:
        # Fallback: use the whole external folder if it directly contains class folders.
        external_test_dir = external_dataset_path

    if not os.path.exists(external_test_dir):
        raise FileNotFoundError(f"External testing path not found: {external_test_dir}")

    external_dataset = datasets.ImageFolder(root=external_test_dir, transform=eval_tf)

    return external_dataset, external_test_dir


def make_dataloader(dataset, shuffle):
    """Create a DataLoader using the local machine settings from Section 3."""
    kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
    return DataLoader(dataset, **kwargs)


def create_dataloaders(train_dataset, valid_dataset, test_dataset, external_dataset=None):
    """Create DataLoader objects for train, validation, test, and optional external test dataset."""

    train_loader = make_dataloader(train_dataset, shuffle=True)
    valid_loader = make_dataloader(valid_dataset, shuffle=False)
    test_loader = make_dataloader(test_dataset, shuffle=False)

    external_loader = None
    if external_dataset is not None:
        external_loader = make_dataloader(external_dataset, shuffle=False)

    return train_loader, valid_loader, test_loader, external_loader


# In[27]:


# ============================================================
# 6. Plotting Helpers
# ============================================================

def denormalize_image(img_tensor):
    """Convert normalized tensor image back to displayable image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1)
    return img


def plot_class_distribution(dataset, class_names, title, save_path=None):
    """Plot and optionally save class distribution."""
    labels = [label for _, label in dataset.samples]
    counts = np.bincount(labels, minlength=len(class_names))

    plt.figure(figsize=(8, 5))
    plt.bar(class_names, counts)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()

    return dict(zip(class_names, counts.astype(int).tolist()))


def plot_training_curves(history, model_name, dataset_name, run_name, save_dir):
    """Save accuracy, loss, and learning-rate curves."""
    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["valid_acc"], marker="o", label="Valid Accuracy")
    plt.title(f"{model_name} | {dataset_name} | {run_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["valid_loss"], marker="o", label="Valid Loss")
    plt.title(f"{model_name} | {dataset_name} | {run_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["lr"], marker="o")
    plt.title(f"{model_name} | {dataset_name} | {run_name} - Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_rate_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()


# In[28]:


# ============================================================
# 7. Model Definitions
# All models are written to work with 32x32, 64x64, and 128x128 images.
# Adaptive pooling avoids hardcoded feature sizes.
# ============================================================

class SimpleCNN(nn.Module):
    """Small baseline CNN."""
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    """Basic residual block used in ResNet-18 and ResNet-34."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """CIFAR-style ResNet-18, suitable for small and medium image sizes."""
    def __init__(self, block, num_classes):
        super(ResNet18, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64,  2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class ResNet34(nn.Module):
    """
    CIFAR-style ResNet-34.

    ResNet-34 uses BasicBlock with layer configuration [3, 4, 6, 3].
    This version uses a 3x3 first convolution and no initial max-pooling,
    which is better for small images such as 32x32, while still working
    for 64x64 and 128x128 images because AdaptiveAvgPool2d is used.
    """
    def __init__(self, block, num_classes):
        super(ResNet34, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet-34 layer configuration: [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 64,  3, stride=1)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        self.layer3 = self._make_layer(block, 256, 6, stride=2)
        self.layer4 = self._make_layer(block, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []

        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


class VGGSmall(nn.Module):
    """Small VGG-style model with adaptive pooling."""
    def __init__(self, num_classes):
        super(VGGSmall, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, stride=stride, padding=1,
                groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MobileNetSmall(nn.Module):
    """Small MobileNet-style model."""
    def __init__(self, num_classes):
        super(MobileNetSmall, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# In[29]:


# ============================================================
# 8. Generic Model Factory and Model Information Saving
# ============================================================

AVAILABLE_MODEL_NAMES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "mobilenetv2",
    "mobilenetv3large",
    "efficientnetb0",
    "efficientnetb1",
    "convnexttiny",
    "vgg16",
    "vgg19",
    "densenet121",
]


def create_torchvision_model(model_builder):
    """Build a torchvision model without downloading pretrained weights."""
    try:
        return model_builder(weights=None)
    except TypeError:
        # Older torchvision versions use pretrained instead of weights.
        return model_builder(pretrained=False)


def create_model(model_name, num_classes):
    """Create selected torchvision model using only its name."""
    model_name = model_name.lower().strip()

    if model_name == "resnet18":
        model = create_torchvision_model(models.resnet18)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet34":
        model = create_torchvision_model(models.resnet34)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = create_torchvision_model(models.resnet50)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenetv2":
        model = create_torchvision_model(models.mobilenet_v2)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == "mobilenetv3large":
        model = create_torchvision_model(models.mobilenet_v3_large)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == "efficientnetb0":
        model = create_torchvision_model(models.efficientnet_b0)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == "efficientnetb1":
        model = create_torchvision_model(models.efficientnet_b1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == "convnexttiny":
        model = create_torchvision_model(models.convnext_tiny)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == "vgg16":
        model = create_torchvision_model(models.vgg16)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == "vgg19":
        model = create_torchvision_model(models.vgg19)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == "densenet121":
        model = create_torchvision_model(models.densenet121)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Available: {', '.join(AVAILABLE_MODEL_NAMES)}"
        )

    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_layer_details(model):
    """Return layer/module details as a pandas DataFrame."""
    rows = []

    for name, module in model.named_modules():
        if name == "":
            continue

        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)

        rows.append({
            "layer_name": name,
            "layer_type": module.__class__.__name__,
            "parameters": int(params),
            "trainable_parameters": int(trainable),
            "details": str(module)
        })

    return pd.DataFrame(rows)


def save_model_information(model, save_dir):
    """Save architecture text, layer details, and parameter counts."""
    total_params, trainable_params = count_parameters(model)

    save_text(model, os.path.join(save_dir, "model_architecture.txt"))

    layer_df = get_layer_details(model)
    layer_df.to_csv(os.path.join(save_dir, "layer_details.csv"), index=False)
    layer_df.to_json(os.path.join(save_dir, "layer_details.json"), orient="records", indent=4)

    param_info = {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params)
    }
    save_json(param_info, os.path.join(save_dir, "parameter_count.json"))

    return param_info


# In[30]:


# ============================================================
# 9. Training and Evaluation Functions
# ============================================================

def train_one_epoch(model, data_loader, criterion, optimizer, device, scaler=None, progress_desc="Train"):
    """Train model for one epoch."""
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    use_amp = scaler is not None and device.type == "cuda"

    progress = tqdm_progress(data_loader, desc=progress_desc, unit="batch", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=PIN_MEMORY)
        labels = labels.to(device, non_blocking=PIN_MEMORY)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

        progress.set_postfix({
            "loss": f"{running_loss / total_samples:.4f}",
            "acc": f"{running_corrects / total_samples:.4f}"
        })

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def evaluate_one_epoch(model, data_loader, criterion, device, progress_desc="Valid"):
    """Evaluate model for one epoch."""
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        progress = tqdm_progress(data_loader, desc=progress_desc, unit="batch", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=PIN_MEMORY)
            labels = labels.to(device, non_blocking=PIN_MEMORY)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

            progress.set_postfix({
                "loss": f"{running_loss / total_samples:.4f}",
                "acc": f"{running_corrects / total_samples:.4f}"
            })

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def get_predictions(model, data_loader, device, progress_desc="Predict"):
    """Return true labels, predicted labels, and prediction probabilities."""
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        progress = tqdm_progress(data_loader, desc=progress_desc, unit="batch", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=PIN_MEMORY)

            outputs = model(images)
            probs = softmax(outputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            progress.set_postfix({"samples": len(all_labels)})

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# In[31]:


# ============================================================
# 10. Evaluation Saving Function
# ============================================================

def evaluate_and_save(model, data_loader, class_names, device, save_dir, prefix):
    """
    Evaluate a model and save metrics, reports, predictions, and plots.
    prefix examples: internal_test, external_test.
    """

    labels, preds, probs = get_predictions(
        model,
        data_loader,
        device,
        progress_desc=f"{prefix} predictions"
    )

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds,
        average="weighted",
        zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1)
    }

    save_json(metrics, os.path.join(save_dir, f"{prefix}_metrics.json"))

    # Use a fixed label list so reports/confusion matrices do not crash
    # when one class is missing from a small test split.
    labels_list = list(range(len(class_names)))

    report_dict = classification_report(
        labels, preds,
        labels=labels_list,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )
    report_text = classification_report(
        labels, preds,
        labels=labels_list,
        target_names=class_names,
        zero_division=0
    )

    save_json(report_dict, os.path.join(save_dir, f"{prefix}_classification_report.json"))
    save_text(report_text, os.path.join(save_dir, f"{prefix}_classification_report.txt"))

    pred_df = pd.DataFrame({
        "true_label_index": labels,
        "predicted_label_index": preds,
        "true_label_name": [class_names[i] for i in labels],
        "predicted_label_name": [class_names[i] for i in preds]
    })

    for i, class_name in enumerate(class_names):
        pred_df[f"prob_{class_name}"] = probs[:, i]

    pred_df.to_csv(os.path.join(save_dir, f"{prefix}_predictions.csv"), index=False)

    cm = confusion_matrix(labels, preds, labels=labels_list)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(save_dir, f"{prefix}_confusion_matrix.csv"))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{prefix} Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    plt.close()

    # Normalized confusion matrix.
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.divide(cm.astype("float"), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation="nearest")
    plt.title(f"{prefix} Normalized Confusion Matrix")
    plt.colorbar()
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, f"{cm_normalized[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_normalized_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    plt.close()

    # Per-class accuracy plot.
    per_class_acc = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros(len(class_names)), where=cm.sum(axis=1) != 0)
    per_class_df = pd.DataFrame({"class_name": class_names, "accuracy": per_class_acc})
    per_class_df.to_csv(os.path.join(save_dir, f"{prefix}_per_class_accuracy.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(class_names, per_class_acc)
    plt.title(f"{prefix} Per-Class Accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_per_class_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    plt.close()

    # ROC curve.
    try:
        num_classes = len(class_names)

        if num_classes == 2:
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(7, 5))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.title(f"{prefix} ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}_roc_curve.png"), dpi=150, bbox_inches="tight")
            plt.close()
            plt.close()

            metrics["auc"] = float(roc_auc)

        else:
            labels_bin = label_binarize(labels, classes=list(range(num_classes)))

            plt.figure(figsize=(8, 6))
            auc_values = {}

            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                auc_values[class_names[i]] = float(roc_auc)
                plt.plot(fpr, tpr, label=f"{class_names[i]} AUC = {roc_auc:.4f}")

            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.title(f"{prefix} Multi-Class ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}_roc_curve.png"), dpi=150, bbox_inches="tight")
            plt.close()
            plt.close()

            metrics["auc_per_class"] = auc_values

        save_json(metrics, os.path.join(save_dir, f"{prefix}_metrics.json"))

    except Exception as e:
        save_text(f"ROC curve could not be created. Reason: {e}", os.path.join(save_dir, f"{prefix}_roc_error.txt"))

    return metrics


# In[32]:


# ============================================================
# 12. One Complete Training Run with Local Auto-Resume
# ============================================================

def train_single_run(dataset_cfg, run_number):
    """
    Train one model run for one dataset size.

    Saving structure in local output directory:
    PROJECT_ROOT/training_results/MODEL_NAME/dataset_name/run_XX/

    Resume logic:
    - If run_completed.json exists and SKIP_COMPLETED_RUNS=True, skip this run.
    - If last_checkpoint.pth exists and RESUME_TRAINING=True, continue from saved epoch.
    - A new last_checkpoint.pth is saved after every epoch.
    """

    dataset_name = dataset_cfg["dataset_name"]
    image_size = dataset_cfg["image_size"]
    main_dataset_path = resolve_existing_path(dataset_cfg["main_dataset_path"])
    external_dataset_path = resolve_existing_path(dataset_cfg["external_dataset_path"])

    run_name = f"run_{run_number:02d}"
    run_dir = os.path.join(RESULTS_ROOT, MODEL_NAME, dataset_name, run_name)
    make_dir(run_dir)

    completed_path = os.path.join(run_dir, "run_completed.json")
    run_summary_path = os.path.join(run_dir, "run_summary.json")
    training_log_path = os.path.join(run_dir, "training_log.txt")
    last_checkpoint_path = os.path.join(run_dir, "last_checkpoint.pth")
    best_model_path = os.path.join(run_dir, "best_model.pth")
    final_model_path = os.path.join(run_dir, "final_model.pth")

    # If this run is already completed, skip it to avoid overwriting useful results.
    if SKIP_COMPLETED_RUNS and os.path.exists(completed_path):
        print(f"Skipping completed run: {run_dir}")
        existing_summary = load_json(run_summary_path, default=None)
        if existing_summary is not None:
            return existing_summary
        return {
            "model_name": MODEL_NAME,
            "dataset_name": dataset_name,
            "image_size": image_size,
            "run_name": run_name,
            "status": "already_completed",
            "run_dir": run_dir
        }

    run_seed = SEED + run_number
    set_seed(run_seed)

    append_text("=" * 80, training_log_path)
    append_text(f"Starting/Resuming: {MODEL_NAME} | {dataset_name} | {run_name}", training_log_path)
    append_text(f"Run folder: {run_dir}", training_log_path)
    append_text(f"Main dataset path: {main_dataset_path}", training_log_path)
    append_text(f"External dataset path: {external_dataset_path}", training_log_path)

    # Load datasets.
    train_dataset, valid_dataset, test_dataset, train_dir, valid_dir, test_dir = load_main_datasets(
        main_dataset_path,
        image_size
    )

    external_dataset, external_test_dir = load_external_test_dataset(
        external_dataset_path,
        image_size
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    # External dataset labels must follow the same class order as the main dataset.
    # This warning helps you catch class-folder naming/order issues early.
    external_classes_match_main = (external_dataset.classes == class_names)
    if not external_classes_match_main:
        warning_msg = (
            "WARNING: External dataset class names/order are different from the main dataset. "
            f"Main classes: {class_names} | External classes: {external_dataset.classes}"
        )
        print(warning_msg)
        append_text(warning_msg, training_log_path)

    # Save dataset information.
    dataset_info = {
        "dataset_name": dataset_name,
        "image_size": image_size,
        "main_dataset_path": main_dataset_path,
        "external_dataset_path": external_dataset_path,
        "train_dir": train_dir,
        "valid_dir": valid_dir,
        "test_dir": test_dir,
        "external_test_dir": external_test_dir,
        "class_names": class_names,
        "external_class_names": external_dataset.classes,
        "external_classes_match_main": external_classes_match_main,
        "num_classes": num_classes,
        "num_train_images": len(train_dataset),
        "num_valid_images": len(valid_dataset),
        "num_test_images": len(test_dataset),
        "num_external_test_images": len(external_dataset)
    }
    save_json(dataset_info, os.path.join(run_dir, "dataset_info.json"))

    # Save class distributions only once, unless files are missing.
    class_dist_path = os.path.join(run_dir, "class_distributions.json")
    if not os.path.exists(class_dist_path):
        train_dist = plot_class_distribution(
            train_dataset, class_names,
            f"{dataset_name} Train Class Distribution",
            save_path=os.path.join(run_dir, "train_class_distribution.png")
        )
        valid_dist = plot_class_distribution(
            valid_dataset, class_names,
            f"{dataset_name} Validation Class Distribution",
            save_path=os.path.join(run_dir, "valid_class_distribution.png")
        )
        test_dist = plot_class_distribution(
            test_dataset, class_names,
            f"{dataset_name} Test Class Distribution",
            save_path=os.path.join(run_dir, "test_class_distribution.png")
        )
        external_dist = plot_class_distribution(
            external_dataset, external_dataset.classes,
            f"{dataset_name} External Test Class Distribution",
            save_path=os.path.join(run_dir, "external_test_class_distribution.png")
        )
        save_json({
            "train_distribution": train_dist,
            "valid_distribution": valid_dist,
            "test_distribution": test_dist,
            "external_test_distribution": external_dist
        }, class_dist_path)

    train_loader, valid_loader, test_loader, external_loader = create_dataloaders(
        train_dataset,
        valid_dataset,
        test_dataset,
        external_dataset
    )

    # Create model, then wrap it for multi-GPU training when available.
    model = create_model(MODEL_NAME, num_classes).to(DEVICE)
    param_info = save_model_information(model, run_dir)
    model = maybe_wrap_data_parallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_MIXED_PRECISION and DEVICE.type == "cuda"))

    run_config = {
        "model_name": MODEL_NAME,
        "dataset_name": dataset_name,
        "image_size": image_size,
        "run_name": run_name,
        "run_seed": run_seed,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "learning_rate": LEARNING_RATE,
        "device": str(DEVICE),
        "use_multi_gpu": bool(USE_MULTI_GPU),
        "use_mixed_precision": bool(USE_MIXED_PRECISION and DEVICE.type == "cuda"),
        "gpu_count": int(GPU_COUNT),
        "cpu_count": int(CPU_COUNT),
        "num_workers": int(NUM_WORKERS),
        "pin_memory": bool(PIN_MEMORY),
        "data_parallel": bool(isinstance(model, nn.DataParallel)),
        "resume_training": RESUME_TRAINING,
        "skip_completed_runs": SKIP_COMPLETED_RUNS,
        "save_checkpoint_every_epoch": SAVE_CHECKPOINT_EVERY_EPOCH,
        "results_root": RESULTS_ROOT,
        "parameter_info": param_info
    }
    save_json(run_config, os.path.join(run_dir, "run_config.json"))

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
        "lr": []
    }
    best_valid_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1

    # Resume from checkpoint if available.
    if RESUME_TRAINING and os.path.exists(last_checkpoint_path):
        print(f"Resuming from checkpoint: {last_checkpoint_path}")
        checkpoint = load_training_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            DEVICE
        )
        history = checkpoint.get("history", history)
        best_valid_acc = float(checkpoint.get("best_valid_acc", 0.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

        append_text(f"Resumed from epoch {start_epoch - 1}", training_log_path)
        append_text(f"Best validation accuracy so far: {best_valid_acc}", training_log_path)
    else:
        append_text("Starting fresh training run.", training_log_path)

    start_time = time.time()

    # Main training loop.
    epoch_progress = tqdm_progress(
        range(start_epoch, MAX_EPOCHS + 1),
        desc=f"{MODEL_NAME} {dataset_name} {run_name} epochs",
        unit="epoch"
    )
    for epoch in epoch_progress:
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
            scaler=scaler,
            progress_desc=f"Epoch {epoch:04d} train"
        )
        valid_loss, valid_acc = evaluate_one_epoch(
            model,
            valid_loader,
            criterion,
            DEVICE,
            progress_desc=f"Epoch {epoch:04d} valid"
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(int(epoch))
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["valid_loss"].append(float(valid_loss))
        history["valid_acc"].append(float(valid_acc))
        history["lr"].append(float(current_lr))

        line = (
            f"Epoch {epoch:04d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f} | "
            f"Best Val Acc: {best_valid_acc:.4f} | "
            f"LR: {current_lr:.8f}"
        )
        print(line)
        append_text(line, training_log_path)

        # Save best model if validation accuracy improved.
        if valid_acc > best_valid_acc:
            best_valid_acc = float(valid_acc)
            best_epoch = int(epoch)
            epochs_without_improvement = 0

            torch.save({
                "model_name": MODEL_NAME,
                "dataset_name": dataset_name,
                "image_size": image_size,
                "run_name": run_name,
                "model_state_dict": get_model_state_dict(model),
                "class_names": class_names,
                "num_classes": num_classes,
                "best_valid_acc": float(best_valid_acc),
                "best_epoch": int(best_epoch),
                "history": history,
                "run_config": run_config
            }, best_model_path)
        else:
            epochs_without_improvement += 1

        epoch_progress.set_postfix({
            "train_acc": f"{train_acc:.4f}",
            "valid_acc": f"{valid_acc:.4f}",
            "best": f"{best_valid_acc:.4f}"
        })

        scheduler.step()

        # Save last checkpoint after every epoch.
        if SAVE_CHECKPOINT_EVERY_EPOCH:
            save_training_checkpoint(
                last_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                history,
                best_valid_acc,
                best_epoch,
                epochs_without_improvement,
                class_names,
                num_classes,
                run_config
            )

        # Save live history after every epoch so plots/data are not lost.
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)
        save_json(history, os.path.join(run_dir, "history.json"))

        live_status = {
            "status": "training",
            "current_epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_valid_acc": float(best_valid_acc),
            "epochs_without_improvement": int(epochs_without_improvement),
            "last_checkpoint_path": last_checkpoint_path
        }
        save_json(live_status, os.path.join(run_dir, "run_status.json"))

        # Early stopping condition.
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            stop_msg = (
                f"Early stopping at epoch {epoch}. "
                f"Validation accuracy did not improve for "
                f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            print(stop_msg)
            append_text(stop_msg, training_log_path)
            break

    training_time_sec = time.time() - start_time

    # Load best weights before final testing.
    if os.path.exists(best_model_path):
        best_checkpoint = torch_load_safe(best_model_path, map_location=DEVICE)
        load_model_state_dict(model, best_checkpoint["model_state_dict"])
    else:
        append_text("Warning: best_model.pth was not found. Using current model weights.", training_log_path)

    # Save final model checkpoint.
    torch.save({
        "model_name": MODEL_NAME,
        "dataset_name": dataset_name,
        "image_size": image_size,
        "run_name": run_name,
        "model_state_dict": get_model_state_dict(model),
        "class_names": class_names,
        "num_classes": num_classes,
        "best_valid_acc": float(best_valid_acc),
        "best_epoch": int(best_epoch),
        "history": history,
        "run_config": run_config
    }, final_model_path)

    # Save final history and curves.
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)
    save_json(history, os.path.join(run_dir, "history.json"))
    plot_training_curves(history, MODEL_NAME, dataset_name, run_name, run_dir)

    # Evaluate on internal test dataset.
    internal_test_metrics = evaluate_and_save(
        model,
        test_loader,
        class_names,
        DEVICE,
        run_dir,
        prefix="internal_test"
    )

    # Evaluate on external test dataset.
    # This assumes the external dataset has the same class meaning.
    external_test_metrics = evaluate_and_save(
        model,
        external_loader,
        class_names,
        DEVICE,
        run_dir,
        prefix="external_test"
    )

    run_summary = {
        "model_name": MODEL_NAME,
        "dataset_name": dataset_name,
        "image_size": image_size,
        "run_name": run_name,
        "run_seed": run_seed,
        "epochs_completed": len(history["epoch"]),
        "last_epoch": int(history["epoch"][-1]) if len(history["epoch"]) > 0 else 0,
        "best_epoch": int(best_epoch),
        "best_valid_acc": float(best_valid_acc),
        "training_time_sec_this_session": float(training_time_sec),
        "training_time_min_this_session": float(training_time_sec / 60),
        "internal_test_metrics": internal_test_metrics,
        "external_test_metrics": external_test_metrics,
        "run_dir": run_dir,
        "last_checkpoint_path": last_checkpoint_path,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path
    }

    save_json(run_summary, run_summary_path)
    save_json({"status": "completed", **run_summary}, completed_path)
    save_json({"status": "completed", **run_summary}, os.path.join(run_dir, "run_status.json"))

    append_text("\nFinal Run Summary:", training_log_path)
    append_text(json.dumps(run_summary, indent=4), training_log_path)

    # Optional ZIP backup after each completed run.
    if CREATE_ZIP_BACKUP_AFTER_EACH_RUN:
        zip_path = os.path.join(BACKUP_ZIP_ROOT, MODEL_NAME, dataset_name, f"{run_name}.zip")
        zip_folder(run_dir, zip_path)
        run_summary["run_zip_backup_path"] = zip_path
        save_json(run_summary, run_summary_path)
        append_text(f"ZIP backup saved at: {zip_path}", training_log_path)

    # Free GPU memory before the next run.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_summary


# In[ ]:


# ============================================================
# 13. Run Full Local Experiment with Resume Support
# 32x32  : NUM_RUNS runs
# 64x64  : NUM_RUNS runs
# 128x128: NUM_RUNS runs
# ============================================================

def run_full_experiment():
    """Run all configured dataset sizes and repetitions for MODEL_NAME."""
    make_dir(RESULTS_ROOT)
    make_dir(BACKUP_ZIP_ROOT)

    # Load previous global summary if it exists.
    global_summary_json = os.path.join(RESULTS_ROOT, "all_runs_summary.json")
    all_run_summaries = load_json(global_summary_json, default=[])

    selected_sizes = set(DATASET_SIZES_TO_RUN)
    selected_dataset_configs = [cfg for cfg in DATASET_CONFIGS if cfg["image_size"] in selected_sizes]

    dataset_progress = tqdm_progress(
        selected_dataset_configs,
        desc="Datasets",
        unit="dataset"
    )
    for dataset_cfg in dataset_progress:
        dataset_name = dataset_cfg["dataset_name"]
        image_size = dataset_cfg["image_size"]
        dataset_progress.set_postfix({"current": dataset_name})

        print("\n" + "=" * 80)
        print(f"Starting dataset: {dataset_name} | Image size: {image_size}x{image_size}")
        print("=" * 80)

        check_dataset_path(dataset_cfg["main_dataset_path"])
        check_dataset_path(dataset_cfg["external_dataset_path"])

        run_progress = tqdm_progress(
            range(1, NUM_RUNS + 1),
            desc=f"{MODEL_NAME} {dataset_name} runs",
            unit="run"
        )
        for run_number in run_progress:
            run_progress.set_postfix({"run": f"{run_number}/{NUM_RUNS}"})
            print("\n" + "-" * 80)
            print(f"Starting/resuming {MODEL_NAME} | {dataset_name} | Run {run_number}/{NUM_RUNS}")
            print("-" * 80)

            try:
                run_summary = train_single_run(dataset_cfg, run_number)

                # Remove older entry of the same model/dataset/run from global summary.
                all_run_summaries = [
                    item for item in all_run_summaries
                    if not (
                        item.get("model_name") == MODEL_NAME and
                        item.get("dataset_name") == dataset_name and
                        item.get("run_name") == f"run_{run_number:02d}"
                    )
                ]
                all_run_summaries.append(run_summary)

                # Save global summary after every run.
                summary_df = pd.DataFrame(all_run_summaries)
                summary_df.to_csv(os.path.join(RESULTS_ROOT, "all_runs_summary.csv"), index=False)
                save_json(all_run_summaries, global_summary_json)

            except Exception as e:
                error_info = {
                    "model_name": MODEL_NAME,
                    "dataset_name": dataset_name,
                    "image_size": image_size,
                    "run_number": run_number,
                    "run_name": f"run_{run_number:02d}",
                    "status": "error",
                    "error": str(e)
                }
                print("ERROR in run:", error_info)
                all_run_summaries.append(error_info)
                save_json(all_run_summaries, global_summary_json)
                pd.DataFrame(all_run_summaries).to_csv(os.path.join(RESULTS_ROOT, "all_runs_summary.csv"), index=False)

    print("\nAll experiments completed or resumed as far as possible.")
    print("Results root:", RESULTS_ROOT)

    if AUTO_ZIP_CURRENT_MODEL_AFTER_TRAINING:
        final_zip_paths = zip_current_model_results()
        if len(final_zip_paths) > 0:
            print("\nFinal ZIP files are ready in the local backup folder.")


# In[ ]:


# ============================================================
# 14. Local Run Note
# ============================================================

# Before training, edit DATASET_ROOT in Section 3.
# Inputs are read from your local dataset folder.
# Results are saved to PROJECT_ROOT/training_results.
# ZIP backups are saved to PROJECT_ROOT/training_result_zips.


# In[ ]:


# ============================================================
# 13. Compare Results Across All Datasets and Runs
# ============================================================

def print_existing_summary_path():
    """Print the existing global summary path, if training has produced one."""
    summary_path = os.path.join(RESULTS_ROOT, "all_runs_summary.csv")

    if os.path.exists(summary_path):
        print("Summary CSV saved at:", summary_path)
    else:
        print("No summary CSV found yet.")


# In[ ]:


# ============================================================
# 14. Generic Model Loading Function
# ============================================================

def load_trained_model(checkpoint_path, device):
    """
    Load any saved model checkpoint created by this notebook.
    It automatically recreates the correct model architecture.
    """

    checkpoint = torch_load_safe(checkpoint_path, map_location=device)

    model_name = checkpoint["model_name"]
    num_classes = checkpoint["num_classes"]

    model = create_model(model_name, num_classes)
    load_model_state_dict(model, checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    print("Model loaded successfully.")
    print("Model name:", model_name)
    print("Dataset name:", checkpoint.get("dataset_name", "N/A"))
    print("Run name:", checkpoint.get("run_name", "N/A"))
    print("Image size:", image_size)
    print("Classes:", class_names)

    return model, class_names, image_size, checkpoint


# In[ ]:


# ============================================================
# 15. Single Image Prediction Function
# ============================================================

def predict_single_image(image_path, model, class_names, image_size, device):
    """Predict one image using any loaded model."""

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        output = model(input_tensor)
        probs = softmax(output)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(f"Prediction: {pred_class}\nConfidence: {confidence:.4f}")
    plt.axis("off")
    plt.close()

    print("Prediction:", pred_class)
    print("Confidence:", confidence)

    return pred_class, confidence


def print_available_models():
    """Print all model names that can be trained with --model."""
    print("Available models:")
    for model_name in AVAILABLE_MODEL_NAMES:
        print(f"  {model_name}")
    print("\nTrain one with:")
    print("  python main.py --model <model_name>")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one configured wildfire classifier model."
    )
    parser.add_argument(
        "--model",
        choices=AVAILABLE_MODEL_NAMES,
        help="Model name to train. Omit this argument to list available models."
    )
    return parser.parse_args()


def main():
    global MODEL_NAME

    args = parse_args()
    if args.model is None:
        print_available_models()
        return

    MODEL_NAME = args.model.lower().strip()
    print_runtime_config()
    print_dataset_configs()
    run_full_experiment()
    print_existing_summary_path()


# In[ ]:


# ============================================================
# 17. Example: Load a Specific Saved Model Later from Local Results
# ============================================================

# Example path after training:
# checkpoint_path = os.path.join(RESULTS_ROOT, "resnet34", "wildfire_128x128", "run_01", "best_model.pth")
# loaded_model, loaded_class_names, loaded_image_size, checkpoint = load_trained_model(checkpoint_path, DEVICE)

# Example single image prediction from the current dataset:
# image_path = os.path.join(EXTERNAL_DATASET_ROOT, "Forest Fire Dataset 128x128", "test", "fire", "fire_0002.jpg")
# predict_single_image(image_path, loaded_model, loaded_class_names, loaded_image_size, DEVICE)


# In[ ]:


# ============================================================
# 18. Zip Only the Current Model Results Folder
# ============================================================

# This cell manually creates/updates one ZIP backup ONLY for the model currently selected in MODEL_NAME.
# The notebook also runs this automatically at the end of Section 13 when
# AUTO_ZIP_CURRENT_MODEL_AFTER_TRAINING = True.

if __name__ == "__main__":
    main()

