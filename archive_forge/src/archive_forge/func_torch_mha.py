from typing import Any, Dict
import torch
import torch.nn as nn
import triton
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components import MultiHeadDispatch
from xformers.components.attention import ScaledDotProduct
def torch_mha():
    y, _ = torch_multi_head(query=q, key=k, value=v)
    if backward:
        torch.norm(y).backward()
    return y