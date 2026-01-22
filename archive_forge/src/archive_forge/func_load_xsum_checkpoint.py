import argparse
import os
from pathlib import Path
import fairseq
import torch
from packaging import version
from torch import nn
from transformers import (
from transformers.utils import logging
def load_xsum_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    sd = torch.load(checkpoint_path, map_location='cpu')
    hub_interface = torch.hub.load('pytorch/fairseq', 'bart.large.cnn').eval()
    hub_interface.model.load_state_dict(sd['model'])
    return hub_interface