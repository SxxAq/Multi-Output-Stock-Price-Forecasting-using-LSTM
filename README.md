# Multi-Output Stock Forecasting with LSTM

LSTM-based multi-output time-series forecasting for NIFTY-50 stock data. The
project is implemented in a modular Python pipeline and also includes an
executed Jupyter notebook with outputs for assignment submission.

**Input:** Last 10 days of 11 stock features  
**Output:** Next 5 days predictions (all 11 features)

## Dataset
[NIFTY-50 Stock Market Data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/) (2000-2021)

This implementation uses `data/HEROMOTOCO.csv` from the Kaggle dataset. The
selected 11 features are:

`Prev Close`, `Open`, `High`, `Low`, `Last`, `Close`, `VWAP`, `Volume`,
`Turnover`, `Deliverable Volume`, `%Deliverble`

## Model Architecture
- **LSTM**: 64 hidden units, 2 layers
- **FC Layer**: Maps LSTM output to future predictions
- **Loss**: MSE
- **Optimizer**: Adam (lr=0.001)
- **Training**: 20 epochs
- **Input window**: 10 days
- **Forecast horizon**: 5 days

## Results

### Metrics
- **MSE**: 0.0593
- **RMSE**: 0.2436
- **MAE**: 0.1906

### Plots
![Forecasting Results](plots/forecasting_results.png)

Four visualizations:
1. **Training Loss** - Shows decreasing loss over epochs
2. **Date-wise Predictions** - Actual vs Predicted with dates
3. **Sample Comparison** - First 10 test samples side-by-side
4. **Error Distribution** - Histogram of prediction errors

The notebook also includes overall metric plots and per-feature metric plots.

## Notebook

The executed notebook is available at:

`notebooks/forecast.ipynb`

It contains the full workflow, saved outputs, MSE/RMSE/MAE results, and
date-wise prediction plots.

## Running the Project

### 1. Install dependencies
```bash
uv sync
```

### 2. Train and evaluate the model
```bash
uv run python main.py
```
This trains the LSTM model, prints MSE/RMSE/MAE, saves the visualization to
`plots/forecasting_results.png`, and saves the trained model to
`models/lstm_forecaster.pth`.

### 3. Open the executed notebook
```bash
uv run jupyter notebook notebooks/forecast.ipynb
```
The notebook already contains saved outputs for submission.

### 4. Make future predictions
```bash
uv run python predict.py
```
Predicts next 5 days and saves to `plots/predictions.csv`

## Project Structure
```
data/
  HEROMOTOCO.csv       -> NIFTY-50 stock CSV used for training/evaluation
notebooks/
  forecast.ipynb       -> Executed notebook
plots/
  forecasting_results.png
  predictions.csv
models/
  lstm_forecaster.pth
src/
  data.py              -> Load, clean, normalize, and create sequences
  model.py             -> LSTM forecaster
  train.py             -> Training loop
  evaluate.py          -> Metrics and visualization
main.py                -> Pipeline orchestration
predict.py             -> Inference script
pyproject.toml         -> Project dependencies
uv.lock                -> Locked dependency versions
```
