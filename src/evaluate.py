import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
    
    y_pred_np = y_pred.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    y_pred_flat = y_pred_np.reshape(-1)
    y_test_flat = y_test_np.reshape(-1)
    
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    
    return y_pred_np, y_test_np, {'mse': mse, 'rmse': rmse, 'mae': mae}

def plot_results(losses, y_pred_np, y_test_np, features, dates=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(losses, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    if dates is not None and len(dates) == len(y_test_np):
        ax.plot(dates, y_test_np[:, 0, 0], label='Actual', linewidth=2, alpha=0.7, marker='o', markersize=3)
        ax.plot(dates, y_pred_np[:, 0, 0], label='Predicted', linewidth=2, alpha=0.7, marker='x', markersize=3)
        ax.set_xlabel('Date')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.plot(y_test_np[:, :, 0].flatten(), label='Actual', linewidth=2, alpha=0.7)
        ax.plot(y_pred_np[:, :, 0].flatten(), label='Predicted', linewidth=2, alpha=0.7)
        ax.set_xlabel('Time Steps')
    
    ax.set_ylabel('Value')
    ax.set_title(f'Predicted vs Actual - {features[0]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    x_pos = np.arange(min(10, len(y_pred_np)))
    width = 0.35
    sample_actual = y_test_np[:10, 0, 0]
    sample_pred = y_pred_np[:10, 0, 0]
    ax.bar(x_pos - width/2, sample_actual, width, label='Actual', alpha=0.8)
    ax.bar(x_pos + width/2, sample_pred, width, label='Predicted', alpha=0.8)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    ax.set_title('First 10 Samples Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    errors = (y_test_np.reshape(-1) - y_pred_np.reshape(-1))
    ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/forecasting_results.png', dpi=150, bbox_inches='tight')
    print("Plots saved to plots/forecasting_results.png")
