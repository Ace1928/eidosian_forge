from contextlib import contextmanager
from typing import Any, List, Tuple, cast
import random
import torch
import time
from torch.utils.benchmark import Timer
def run_nnc(ir, inputs, dynamic) -> float:
    try:
        strat = [('DYNAMIC', 10)] if dynamic else [('STATIC', 10)]
        old_strat = torch.jit.set_fusion_strategy(strat)
        with torch.jit.fuser('fuser1'):
            return run_test(ir, inputs)
    finally:
        torch.jit.set_fusion_strategy(old_strat)