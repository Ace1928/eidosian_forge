from typing import Any, Dict
import torch
import triton
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components.reversible import ReversibleSequence
def normal_step():
    y = a
    for _ in range(depth):
        y = y + f(y)
        y = y + g(y)
    if backward:
        torch.norm(y).backward()
    return y