import torch
from torch import nn


class RBFLayer(torch.nn.Module):
    def __init__(self, feature_size: int, hidden_size: int):
        super(RBFLayer, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.centers = nn.Parameter(torch.Tensor(hidden_size, feature_size))
        self.sigmas = nn.Parameter(torch.Tensor(hidden_size))
        self.init_weights()

    def forward(self, input: torch.Tensor):
        # repeat each row
        c = self.centers.repeat(1, input.size(0)).reshape(-1, self.feature_size)
        # repeat input
        x = input.repeat(self.hidden_size, 1)
        distances = (
            torch.norm(c - x, dim=1)
            .reshape(self.hidden_size, input.size(0))
            .transpose(1, 0)
        )
        phi = torch.exp(-distances.pow(2) / (2 * self.sigmas.pow(2)))
        return phi

    def init_weights(self):
        self.centers.data.uniform_(0, 1)
        self.sigmas.data.uniform_(0, 1)
