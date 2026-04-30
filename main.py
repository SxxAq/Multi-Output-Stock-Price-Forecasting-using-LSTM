import torch
from pathlib import Path
from src.data import load_and_preprocess, get_features, normalize_data, create_sequences, prepare_tensors
from src.model import LSTMForecaster
from src.train import train_model
from src.evaluate import evaluate, plot_results

Path("plots").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

print("Loading data...")
df = load_and_preprocess("data/HEROMOTOCO.csv")
features = get_features(df)

print("Normalizing...")
df_scaled, scaler = normalize_data(df, features)

print("Creating sequences...")
X, y = create_sequences(df_scaled[features].values)

print("Preparing tensors...")
X_train, X_test, y_train, y_test = prepare_tensors(X, y)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

print("\nTraining...")
model = LSTMForecaster(input_size=len(features), output_steps=5)
losses = train_model(model, X_train, y_train, epochs=20)

print("\nEvaluating...")
y_pred_np, y_test_np, metrics = evaluate(model, X_test, y_test)
print(f"MSE: {metrics['mse']:.6f}")
print(f"RMSE: {metrics['rmse']:.6f}")
print(f"MAE: {metrics['mae']:.6f}")

print("\nPlotting...")
plot_results(losses, y_pred_np, y_test_np, features)

torch.save(model.state_dict(), "models/lstm_forecaster.pth")
print("Model saved to models/lstm_forecaster.pth")
