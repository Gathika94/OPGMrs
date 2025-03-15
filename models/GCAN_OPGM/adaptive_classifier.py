import torch
import torch.nn as nn

class AdaptiveClassifier(nn.Module):
    def __init__(self):
        super(AdaptiveClassifier, self).__init__()
        # Redefine layers for processing only maximum statistics
        self.row_network = nn.Sequential(
            nn.Linear(1, 128),  # Now input is just 1 stat (max)
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.col_network = nn.Sequential(
            nn.Linear(1, 128),  # Now input is just 1 stat (max)
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Assuming x has shape [batch_size, num_rows, num_cols]
        batch_size, num_rows, num_cols = x.shape

        # Process row-wise and column-wise maximums only
        row_max = torch.max(x, dim=2).values.unsqueeze(-1)  # Keep dimensions for linear layer
        col_max = torch.max(x, dim=1).values.unsqueeze(-1)  # Keep dimensions for linear layer

        # Compute predictions using the maximum statistic
        rows_preds = self.row_network(row_max).view(batch_size, 1, -1)
        cols_preds = self.col_network(col_max).view(batch_size, 1, -1)
        
        return rows_preds, cols_preds
