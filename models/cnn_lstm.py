"""
CNN-BiLSTM model for Speech Emotion Recognition.

Input:  (batch, time, features)   e.g. (B, 173, 120)
Output: (batch, num_classes)      logits
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Simple additive attention over the LSTM's time dimension."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, H)
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)   # (B, H)
        return context, weights


class CNN_BiLSTM_SER(nn.Module):
    def __init__(self, input_dim=120, num_classes=7, cnn_channels=(128, 256),
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.attention = Attention(lstm_hidden * 2)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T, F) -> conv1d expects (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)          # (B, C, T')
        x = x.permute(0, 2, 1)          # (B, T', C)
        x, _ = self.lstm(x)             # (B, T', 2*H)
        context, attn_weights = self.attention(x)
        logits = self.classifier(context)
        return logits
