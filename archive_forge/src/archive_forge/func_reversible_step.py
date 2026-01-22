from typing import Any, Dict
import torch
import triton
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components.reversible import ReversibleSequence
def reversible_step():
    y = revseq(b)
    if backward:
        torch.norm(y).backward()
    return y