from contextlib import contextmanager
from typing import Any, List, Tuple, cast
import random
import torch
import time
from torch.utils.benchmark import Timer
def run_baseline_no_fusion(ir, inputs) -> float:
    with no_fuser():
        return run_test(ir, inputs)