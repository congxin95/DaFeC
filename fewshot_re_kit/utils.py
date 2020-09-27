import torch
from torch import nn

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if not m.bias is None:
            nn.init.zeros_(m.bias)
        print("xavier init")
