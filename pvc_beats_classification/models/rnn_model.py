import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers, drop_prob=0.5):
        super(BidirectionalLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=drop_prob, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(2 * hidden_dim, output_size)

    def forward(self, x):
        input = x.permute(1, 0, 2)
        output, (hidden, cell) = self.lstm(input)
        out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        dense_outputs = self.fc(out)
        return dense_outputs#, hidden


if __name__ == "__main__":

    data = torch.rand((40, 1000, 12))
    net = BidirectionalLSTM(12, 9, 100, 2)
