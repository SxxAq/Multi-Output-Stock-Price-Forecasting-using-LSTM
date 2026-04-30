import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, lstm_hidden=64, num_layers=2, output_steps=5):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, input_size * output_steps)
        self.output_steps = output_steps
        self.input_size = input_size
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        fc_out = self.fc(last_output)
        output = fc_out.reshape(-1, self.output_steps, self.input_size)
        return output
