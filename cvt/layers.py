import torch
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# CvT Feed Forward Layer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        ).to(device)
    def forward(self, x):
        return self.nn(x)

# Custom Residual layer
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # allows to pass variable # of params to fn
        x = fn(x, **kwargs) + x
        return x

# Custom Layer Norm
class Norm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim).to(device)
        self.fn = fn
    def forward(self, x, **kwargs):
        # apply layer norm prior to fn
        x = self.norm(x)
        return self.fn(x, **kwargs)