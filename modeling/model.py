import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaModel(nn.Module):
    
    def __init__(self, in_dim, out_dim=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=128,
            num_layers=2
        )
        self.predict = nn.Linear(
            in_features=128,
            out_features=out_dim
        )

    def forward(self, batched_inputs):
        seq_data = batched_inputs["inputs"]
        labels = batched_inputs["labels"]

        seq_data = seq_data.transpose(0, 1)
        labels = labels.transpose(0, 1)

        output, _ = self.encoder(seq_data)

        out = self.predict(output)

        # build loss function
        # e.g. loss = torch.abs(labels - out)
        loss = None

        return loss
