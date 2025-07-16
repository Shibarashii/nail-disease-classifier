import torch
import torchvision
import torchmetrics
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch import nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image

from timeit import default_timer as timer
from tqdm.auto import tqdm
from pathlib import Path
from typing import Union
import json
import os


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    """Performs training with model trying to learn on `data_loader`"""

    train_loss, train_acc = 0, 0

    model.train()
    for batch, (X_train, y_train) in enumerate(data_loader):

        # Put data on target device
        X_train, y_train = X_train.to(device), y_train.to(device)

        # Forward pass (outputs the raw logits from the model)
        y_pred = model(X_train)

        loss = criterion(y_pred, y_train)
        train_loss += loss
        train_acc += accuracy_fn(y_train, y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(
                f"Looked at {batch * len(X_train)} / {len(data_loader.dataset)} samples")

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc


def valid_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               accuracy_fn,
               device: torch.device):
    """Performs validation with model trying to learn on `data_loader`"""

    valid_loss, valid_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X_valid, y_valid in data_loader:
            X_valid, y_valid = X_valid.to(device), y_valid.to(device)

            valid_pred = model(X_valid)

            valid_loss += criterion(valid_pred, y_valid)
            valid_acc += accuracy_fn(y_valid, valid_pred.argmax(dim=1))

        valid_loss /= len(data_loader)
        valid_acc /= len(data_loader)
    return valid_loss, valid_acc


def train_model(epochs: int,
                model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                valid_dataloader: torch.utils.data.DataLoader,
                criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                accuracy_fn,
                device: torch.device):

    # Create  empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "valid_loss": [],
               "valid_acc": []}

    start_time = timer()

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model, train_dataloader, criterion, optimizer, accuracy_fn, device)
        test_loss, test_acc = valid_step(
            model, valid_dataloader, criterion, accuracy_fn, device)

        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f}, Train accuracy : {train_acc:.4f} | Valid loss: {test_loss:.4f}, Valid accuracy: {test_acc:.4f}")

        # Update results dictionary
        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc.item())
        results["valid_loss"].append(test_loss.item())
        results["valid_acc"].append(test_acc.item())

    end_time = timer()
    training_time = end_time - start_time

    print(f"Total time on {device}: {training_time:.3f}")

    return results


def save_results(dir_name: str, file_name: str, results):
    results_dir = Path(dir_name)
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"{file_name}.json"

    # Load existing results if file exists
    if results_file.exists():
        with open(results_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Generate new run key (e.g., "run_1", "run_2", etc.)
    run_id = f"run_{len(all_results) + 1}"
    all_results[run_id] = results

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=4)

    return results_file


def load_results(dir_name: str, file_name: str, run: Union[str, int] = "latest") -> pd.DataFrame:
    """
    Load a specific training run from a results file and return it as a DataFrame.

    Args:
        dir_name (str): Directory containing the results file.
        file_name (str): Name of the JSON file (without `.json`).
        run (str or int): Which run to load. Options:
                          - "latest" (default): most recent run
                          - "all": returns the entire dictionary
                          - int: specific run number, e.g., 2 → "run_2"

    Returns:
        pd.DataFrame: DataFrame of the selected run (or full dict if run='all')
    """
    path = Path(dir_name) / f"{file_name}.json"

    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    with open(path, "r") as f:
        all_runs = json.load(f)

    if run == "all":
        return all_runs  # return raw dict
    elif run == "latest":
        run_key = max(all_runs, key=lambda k: int(k.split("_")[1]))
    else:
        run_key = f"run_{run}"

    return pd.DataFrame(all_runs[run_key])


def predict_compare(test_data: torchvision.datasets.ImageFolder,
                    model: torch.nn.Module,
                    device: torch.device,
                    class_names,
                    random_seed: int = None):
    """
    Makes random predictions and compares it to the ground truth.
    The results will be plotted.
    """
    if random_seed:
        random.seed(random_seed)

    test_samples = []
    test_labels = []

    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    # Make predictions
    pred_probs = []

    model.eval()
    with torch.inference_mode():
        for sample in test_samples:
            sample = torch.unsqueeze(sample, dim=0).to(
                device)  # Add batch dimension
            logit = model(sample)
            pred = torch.softmax(logit.squeeze(), dim=0)
            pred_probs.append(pred.cpu())

    pred_probs = torch.stack(pred_probs)
    pred_classes = torch.argmax(pred_probs, dim=1)

    # Plotting and Comparing prediction to ground truth
    plt.figure(figsize=(10, 7))

    rows, cols = 3, 3

    for i, sample in enumerate(test_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(sample.squeeze(dim=0).permute(1, 2, 0), cmap="gray")

        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]]
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"

        if pred_label == truth_label:
            # green text if prediction is correct
            plt.title(title_text, fontsize=10, c="g")
        else:
            # red text if prediction is incorrect
            plt.title(title_text, fontsize=10, c="r")

        plt.axis(False)


def make_predictions(model: torch.nn.Module,
                     test_dataloader: torch.utils.data.DataLoader,
                     test_data: torchvision.datasets.ImageFolder,
                     device: torch.device):
    """Makes predictions and returns `y_preds` and `y_true`"""
    y_preds = []

    model.eval()
    with torch.inference_mode():
        for X, y, in tqdm(test_dataloader, desc="Making predictions"):
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            y_pred = torch.softmax(y_logits.squeeze(), dim=1).argmax(dim=1)
            y_preds.append(y_pred.cpu())

    y_preds = torch.cat(y_preds)
    y_true = torch.tensor(test_data.targets)

    return y_preds, y_true


def make_single_prediction(model: torch.nn.Module,
                           image_path: Path,
                           transforms: torchvision.transforms,
                           device: torch.device):
    if not transforms:
        # Normalization and transforming data into tensors
        transforms = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms(image).unsqueeze(0).to(
        device)  # Add batch dim and move to device

    model.eval()
    with torch.inference_mode():
        output = model(image_tensor)  # logits
        pred_prob = torch.softmax(output, dim=1).squeeze(0)
        pred_class = pred_prob.argmax().item()

    return pred_prob, pred_class
