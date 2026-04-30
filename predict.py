import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.data import load_and_preprocess, get_features, normalize_data
from src.model import LSTMForecaster

model_path = Path("models/lstm_forecaster.pth")
if not model_path.exists():
    print("❌ Model not found. Train first: python main.py")
    exit(1)

print("Loading model...")
df = load_and_preprocess("data/HEROMOTOCO.csv")
features = get_features(df)
df_scaled, scaler = normalize_data(df, features)

num_features = len(features)
model = LSTMForecaster(input_size=num_features, output_steps=5)
model.load_state_dict(torch.load(model_path))
model.eval()

print("\nMaking prediction on latest 10 days...")
last_10_days = df_scaled[features].values[-10:]
X_input = torch.FloatTensor(last_10_days).unsqueeze(0)

with torch.no_grad():
    predictions_normalized = model(X_input).squeeze(0).numpy()

predictions = scaler.inverse_transform(predictions_normalized)

print("\n📊 Next 5 Days Predictions:")
print("-" * 80)

last_date = df['Date'].iloc[-1]
for day in range(5):
    pred_date = last_date + pd.Timedelta(days=day+1)
    print(f"\n{pred_date.strftime('%Y-%m-%d')}:")
    for feat_idx, feat in enumerate(features):
        print(f"  {feat}: {predictions[day, feat_idx]:.2f}")

pred_df = pd.DataFrame(
    predictions,
    columns=features,
    index=pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5)
)
pred_df.to_csv("plots/predictions.csv")
print(f"\n✓ Predictions saved to plots/predictions.csv")
print(f"\nPredictions DataFrame:")
print(pred_df)
