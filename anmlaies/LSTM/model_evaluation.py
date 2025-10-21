import os, sys, json, logging
import pandas as pd, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import joblib
from sqlalchemy import create_engine
from urllib.parse import urlparse, quote_plus
from leap.utils.Utilities import Utilities
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logging.info("=== Script started ===")

# ---------------- Config ----------------
HOST_FILTER = "HOST-115AF0E56D335A15"
START, END = "2025-10-06 00:00:00", "2025-10-07 00:00:00"
WINDOW, HORIZON = 60, 10
EPOCHS, BATCH_SIZE, LR = 50, 32, 1e-3
TARGET_METRIC = "Read Ops"
SAVE_DIR = "/usr/lstm_model/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- Reproducibility ----------------
torch.manual_seed(42)
np.random.seed(42)

# ---------------- DB Connection ----------------
def get_db_connection(args):
    db_cfg = json.loads(args['mysqlDS'])
    db_pwd = Utilities.decrypt(db_cfg['password'], db_cfg['salt'])
    parsed = urlparse(db_cfg['url'][5:])
    DB_USER, DB_PWD = db_cfg['userName'], db_pwd
    DB_HOST, DB_PORT, DB_NAME = parsed.hostname, parsed.port, parsed.path.lstrip('/')
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:%s@{DB_HOST}:{DB_PORT}/{DB_NAME}"
                           % quote_plus(DB_PWD))
    return engine

args_dict = {}
for arg in sys.argv:
    try: k,v = arg.split(':',1); args_dict[k]=v
    except: pass

logging.info("Connecting to database...")
engine = get_db_connection(args_dict)
logging.info("Database connection established.")

# ---------------- SQL ----------------
query = f"""
SELECT host,timestamp,metricType,
       AVG(value) AS metric_value
FROM dynatrace_selnav_db
WHERE host = '{HOST_FILTER}'
  AND metricType = '{TARGET_METRIC}'
  AND timestamp >= '{START}'
  AND timestamp <  '{END}'
GROUP BY host,timestamp,metricType
ORDER BY host,timestamp"""
logging.info("Running SQL query...")
df = pd.read_sql(query, engine)
df['timestamp'] = pd.to_datetime(df['timestamp'])
logging.info(f"Data pulled: {len(df)} rows")

# ---------------- Pivot + Resample ----------------
df_wide = df.pivot_table(index=['timestamp'], columns='metricType', values='metric_value')
g = df_wide.resample('1min').mean().ffill().reset_index()
logging.info(f"Resampled shape: {g.shape}")

# ---------------- Feature Engineering ----------------
g['hour'] = g['timestamp'].dt.hour
g['dayofweek'] = g['timestamp'].dt.dayofweek
g['hour_sin'] = np.sin(2*np.pi*g['hour']/24)
g['hour_cos'] = np.cos(2*np.pi*g['hour']/24)
g['dow_sin'] = np.sin(2*np.pi*g['dayofweek']/7)
g['dow_cos'] = np.cos(2*np.pi*g['dayofweek']/7)

feature_cols = ['hour_sin','hour_cos','dow_sin','dow_cos']
target_cols = [TARGET_METRIC]

# ---------------- Split ----------------
train_df = g[g['timestamp'] < '2025-10-06 18:00:00']
test_df  = g[g['timestamp'] >= '2025-10-06 18:00:00']

# Scale using only train
scaler = StandardScaler()
train_df.loc[:, target_cols] = scaler.fit_transform(train_df[target_cols])
test_df.loc[:, target_cols]  = scaler.transform(test_df[target_cols])
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.joblib"))
logging.info("Scaling complete.")

logging.info(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

# ---------------- Dataset ----------------
class SequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, window, horizon):
        self.X = df[feature_cols+target_cols].values.astype(np.float32)
        self.y = df[target_cols].values.astype(np.float32)
        self.window, self.horizon = window, horizon
        self.max_index = max(0, len(df)-window-horizon+1)
    def __len__(self): return self.max_index
    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx:idx+self.window]),
                torch.from_numpy(self.y[idx+self.window+self.horizon-1]))

train_ds = SequenceDataset(train_df, feature_cols, target_cols, WINDOW, HORIZON)
test_ds  = SequenceDataset(test_df, feature_cols, target_cols, WINDOW, HORIZON)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
logging.info(f"Training samples: {len(train_ds)}, Testing samples: {len(test_ds)}")

# ---------------- Model ----------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, 1)
    def forward(self, x):
        out,_ = self.lstm1(x); out,_ = self.lstm2(out)
        return self.fc(self.drop(out[:,-1,:]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(len(feature_cols)+1).to(device)
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
logging.info(f"Model initialized:\n{model}")

# ---------------- Training with Early Stopping ----------------
best_val_loss = float("inf")
patience, patience_counter = 5, 0

for epoch in range(EPOCHS):
    model.train(); epoch_loss = 0
    for xb,yb in train_loader:
        xb,yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= max(1,len(train_loader))
    logging.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss={epoch_loss:.4f}")

    # Early stopping check on test set
    model.eval(); val_loss = 0
    with torch.no_grad():
        for xb,yb in test_loader:
            preds = model(xb.to(device)).squeeze()
            val_loss += criterion(preds, yb.to(device).squeeze()).item()
    val_loss /= max(1,len(test_loader))
    logging.info(f"Validation Loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            logging.info("Early stopping triggered.")
            break

# ---------------- Evaluation ----------------
logging.info("Starting evaluation...")
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model.pt")))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb,yb in test_loader:
        preds = model(xb.to(device)).squeeze()
        y_true.append(yb.numpy())
        y_pred.append(preds.cpu().numpy())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)
y_true_orig = scaler.inverse_transform(y_true.reshape(-1,1)).ravel()
y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1,1)).ravel()

rmse = mean_squared_error(y_true_orig, y_pred_orig, squared=False)
mae = mean_absolute_error(y_true_orig, y_pred_orig)
r2 = r2_score(y_true_orig, y_pred_orig)
logging.info(f"{TARGET_METRIC}: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")
# ---------------- Plot ----------------
test_times = test_df['timestamp'].iloc[WINDOW+HORIZON-1 : WINDOW+HORIZON-1 + len(y_true_orig)]

plt.figure(figsize=(12,5))
plt.plot(test_times, y_true_orig, label="Actual")
plt.plot(test_times, y_pred_orig, label="Predicted")
plt.title(f"{TARGET_METRIC} Prediction")
plt.xlabel("Time")
plt.ylabel(TARGET_METRIC)
plt.legend()
plt.tight_layout()

plot_path = os.path.join(SAVE_DIR, "read_ops.png")
plt.savefig(plot_path)
plt.close()

logging.info(f"Plot saved to {plot_path}")
logging.info("=== Script finished ===")

