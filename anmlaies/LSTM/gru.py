import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump
import matplotlib.pyplot as plt
import time
import logging
import os

# ---------------- CONFIG ----------------
CSV_PATH = "training_dataset.csv"
SEQ_LEN = 60
PRED_HORIZON = 5
BATCH_SIZE = 64
EPOCHS = 40
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("models_gru", exist_ok=True)

# ---------------- LOGGING ----------------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logging.info(f"Running on device: {DEVICE}")

# ---------------- DATASET ----------------
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.X = sequences
        self.y = targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# ---------------- MODEL ----------------
class GRUModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# ---------------- UTILS ----------------
def create_sequences(data, seq_len, pred_horizon):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_horizon + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len+pred_horizon-1, 0])
    return np.array(X), np.array(y)

def plot_predictions(actual, predicted, metric_name):
    plt.figure(figsize=(10,5))
    plt.plot(actual, label="Actual", linewidth=1.5)
    plt.plot(predicted, label="Predicted", linewidth=1.5)
    plt.xlabel("Time Steps")
    plt.ylabel(metric_name)
    plt.title(f"Predicted vs Actual - {metric_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"pred_vs_actual_{metric_name.replace(' ', '_')}.png")
    plt.close()

# ---------------- MAIN ----------------
start_time = time.time()
df = pd.read_csv(CSV_PATH)
df = df.drop(columns=['Unnamed: 0'])
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Add simple time features
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek

metrics = [col for col in df.columns if col not in ["timestamp", "hour", "dayofweek"]]

for METRIC in metrics:
    logging.info(f"🚀 Starting GRU training for metric: {METRIC}")

    features = df[[METRIC, "hour", "dayofweek"]].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    dump(scaler, f"models_gru/scaler_{METRIC.replace(' ', '_')}.joblib")

    X, y = create_sequences(scaled, SEQ_LEN, PRED_HORIZON)
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = GRUModel(input_size=3).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                output = model(X_batch)
                loss = criterion(output, y_batch.unsqueeze(1))
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"{METRIC} - Epoch [{epoch+1}/{EPOCHS}] - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"models_gru/model_{METRIC.replace(' ', '_')}.pt")

    # ----- Evaluation -----
    model.load_state_dict(torch.load(f"models_gru/model_{METRIC.replace(' ', '_')}.pt"))
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        preds = model(X_test_tensor).cpu().numpy()

    preds_full = np.zeros((preds.shape[0], features.shape[1]))
    preds_full[:, 0] = preds[:, 0]
    actual_full = np.zeros((len(y_test), features.shape[1]))
    actual_full[:, 0] = y_test

    preds_inv = scaler.inverse_transform(preds_full)[:, 0]
    actual_inv = scaler.inverse_transform(actual_full)[:, 0]

    rmse = np.sqrt(mean_squared_error(actual_inv, preds_inv))
    mae = mean_absolute_error(actual_inv, preds_inv)
    r2 = r2_score(actual_inv, preds_inv)

    logging.info(f"✅ {METRIC} - Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    plot_predictions(actual_inv, preds_inv, METRIC)

logging.info(f"🏁 All metrics trained successfully in {(time.time() - start_time):.2f} seconds.")
