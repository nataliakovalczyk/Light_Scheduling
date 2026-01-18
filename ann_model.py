# ann_model.py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class ActivityPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.2):
        super(ActivityPredictor, self).__init__()

        self.model = nn.Sequential(
            # input layer
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # second hidden layer - half the neurons
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # output layer
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.model(x)


def prepare_ann_data(
    dataset: pd.DataFrame,
    feature_columns=None,
    target_column: str = "activity_index",
    train_split: float = 0.8,
):
    """
    Matches your Colab preprocessing:
    - chronological split
    - ColumnTransformer:
        hour, day_of_week -> MinMaxScaler
        temperature -> StandardScaler
        rain, fog -> passthrough
    Returns:
      preprocessor, X_train, X_val, y_train, y_val
    """
    if feature_columns is None:
        feature_columns = ["hour", "day_of_week", "temperature", "rain", "fog"]

    X = dataset[feature_columns].copy()
    y = dataset[target_column].copy()

    # splitting the data chronologically
    split_index = int(train_split * len(X))
    X_train_raw = X.iloc[:split_index].copy()
    X_val_raw = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].values
    y_val = y.iloc[split_index:].values

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_bounded', MinMaxScaler(), ['hour', 'day_of_week']),
            ('num_continuous', StandardScaler(), ['temperature']),
            ('binary', 'passthrough', ['rain', 'fog'])
        ],
        verbose_feature_names_out=False
    )

    preprocessor.fit(X_train_raw)

    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    return preprocessor, X_train, X_val, y_train, y_val


def make_dataloaders(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size: int = 64,
    shuffle_train: bool = True,
):
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Note: your comment says "Changed shuffle=False", but your code uses shuffle=True.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor


def compare_optimizers(
    input_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    X_val_tensor: torch.Tensor,
    y_val_tensor: torch.Tensor,
    n_epochs: int = 50,
    optimizer_candidates=None,
):
    """
    Replicates your Colab optimizer comparison loop and plots losses.
    Returns:
      results_data (list of dicts), best_optimizer_name (str)
    """
    if optimizer_candidates is None:
        optimizer_candidates = ['Adam', 'RMSprop', 'SGD', 'AdamW']

    results_data = []

    plt.figure(figsize=(15, 10))
    print(f"Rozpoczynam szczegółowe porównanie ({n_epochs} epok)...")
    print("-" * 80)
    print(f"{'Optimizer':<12} | {'Min Val Loss':<12} | {'Final MSE':<12} | {'Final R2':<10}")
    print("-" * 80)

    best_val_loss = float('inf')
    best_optimizer_name = 'Adam'

    for i, opt_name in enumerate(optimizer_candidates):
        model = ActivityPredictor(input_dim)
        criterion = nn.MSELoss()

        if opt_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        elif opt_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=0.001)
        elif opt_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        elif opt_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=0.001)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            model.train()
            batch_losses = []
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                pred = model(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            train_losses.append(np.mean(batch_losses))

            model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    pred = model(X_b)
                    loss = criterion(pred, y_b)
                    val_batch_losses.append(loss.item())
            val_losses.append(np.mean(val_batch_losses))

        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(X_val_tensor)
            y_pred = y_pred_tensor.cpu().numpy().flatten()
            y_true = y_val_tensor.cpu().numpy().flatten()

        final_mse = mean_squared_error(y_true, y_pred)
        final_r2 = r2_score(y_true, y_pred)
        min_loss = min(val_losses)

        results_data.append({
            'name': opt_name,
            'mse': final_mse,
            'r2': final_r2,
            'min_loss': min_loss
        })

        print(f"{opt_name:<12} | {min_loss:.5f}      | {final_mse:.5f}      | {final_r2:.4f}")

        if min_loss < best_val_loss:
            best_val_loss = min_loss
            best_optimizer_name = opt_name

        plt.subplot(2, 2, i + 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Val', linestyle='--')
        plt.title(f"{opt_name}\nMSE: {final_mse:.4f} | R2: {final_r2:.3f}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("optimizer_comparison.png")
    plt.show()

    return results_data, best_optimizer_name


def train_final_model(
    input_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    X_val_tensor: torch.Tensor,
    y_val_tensor: torch.Tensor,
    n_epochs: int = 50,
):
    """
    Matches your Colab final training:
    - Adam lr=0.001
    - ReduceLROnPlateau
    Returns:
      final_model, history(dict), metrics(dict), (y_true, y_pred)
    """
    final_model = ActivityPredictor(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(n_epochs):
        final_model.train()
        running_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = final_model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_b.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)

        final_model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                pred = final_model(X_b)
                loss = criterion(pred, y_b)
                running_val_loss += loss.item() * X_b.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f}")

    print("Trening zakończony.")

    final_model.eval()
    with torch.no_grad():
        y_pred_tensor = final_model(X_val_tensor)
        y_pred = y_pred_tensor.cpu().numpy().flatten()
        y_true = y_val_tensor.cpu().numpy().flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("\n" + "=" * 40)
    print("EVALUATION")
    print("=" * 40)
    print(f"MAE (Mean Error):       {mae:.4f}")
    print(f"RMSE (Error Squared): {rmse:.4f}")
    print(f"R2 Score:   {r2:.4f}")

    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    return final_model, history, metrics, (y_true, y_pred)


def plot_training_and_predictions(history, y_true, y_pred):
    """
    Replicates your Colab evaluation plots:
    - learning curve
    - scatter true vs pred
    """
    plt.figure(figsize=(18, 5))

    # Learning Curve
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Scatter Plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal Fit')
    plt.title(f' (R2={r2_score(y_true, y_pred):.2f})')
    plt.xlabel('Real activity')
    plt.ylabel('Predicted activity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_ann_sample(final_model, X_val_tensor, y_val, n_hours: int = 48):
    """
    Replicates your last ANN sample plot.
    Note: your Colab code accidentally used `model` instead of `final_model` here;
    this function uses the trained `final_model`.
    """
    final_model.eval()
    with torch.no_grad():
        sample_predictions = final_model(X_val_tensor[:n_hours]).cpu().numpy().flatten()

    plt.figure(figsize=(10, 4))
    plt.plot(y_val[:n_hours], label="True Activity")
    plt.plot(sample_predictions, label="Predicted Activity")
    plt.title("ANN Activity Prediction (Validation Sample)")
    plt.xlabel("Hour")
    plt.ylabel("Activity Index")
    plt.legend()
    plt.grid(True)
    plt.show()


def run_ann_pipeline(
    csv_path: str = "street_light_dataset.csv",
    batch_size: int = 64,
    n_epochs_compare: int = 50,
    n_epochs_final: int = 50,
):
    """
    Convenience wrapper to run the full ANN section end-to-end.
    Returns:
      dict with preprocessor, final_model, history, metrics, etc.
    """
    dataset = pd.read_csv(csv_path)

    preprocessor, X_train, X_val, y_train, y_val = prepare_ann_data(dataset)
    train_loader, val_loader, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor = make_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=batch_size, shuffle_train=True
    )

    input_dim = X_train.shape[1]
    print(f"Model input dimension: {input_dim}")

    results_data, best_optimizer_name = compare_optimizers(
        input_dim=input_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        X_val_tensor=X_val_tensor,
        y_val_tensor=y_val_tensor,
        n_epochs=n_epochs_compare,
    )

    final_model, history, metrics, (y_true, y_pred) = train_final_model(
        input_dim=input_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        X_val_tensor=X_val_tensor,
        y_val_tensor=y_val_tensor,
        n_epochs=n_epochs_final,
    )

    plot_training_and_predictions(history, y_true, y_pred)
    plot_ann_sample(final_model, X_val_tensor, y_val, n_hours=48)

    return {
        "dataset": dataset,
        "preprocessor": preprocessor,
        "results_data": results_data,
        "best_optimizer_name": best_optimizer_name,
        "final_model": final_model,
        "history": history,
        "metrics": metrics,
        "X_val_tensor": X_val_tensor,
        "y_val": y_val,
    }


if __name__ == "__main__":
    run_ann_pipeline()
