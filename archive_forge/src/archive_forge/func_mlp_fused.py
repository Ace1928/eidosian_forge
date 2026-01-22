import argparse
from typing import Any, Dict
import torch
import triton
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components import Activation
from xformers.components.feedforward import MLP, FusedMLP
def mlp_fused():
    y = fused_mlp(a)
    if backward:
        torch.norm(y).backward()
    return y