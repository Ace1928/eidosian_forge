import argparse
import struct
import torch
from typing import Dict
def get_layer_count(state_dict: Dict[str, torch.Tensor]) -> int:
    n_layer: int = 0
    while f'blocks.{n_layer}.ln1.weight' in state_dict:
        n_layer += 1
    assert n_layer > 0
    return n_layer