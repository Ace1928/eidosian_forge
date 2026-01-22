from contextlib import contextmanager
from typing import Any, List, Tuple, cast
import random
import torch
import time
from torch.utils.benchmark import Timer
@contextmanager
def no_fuser(*args, **kwargs):
    old_optimize = torch._C._get_graph_executor_optimize(False)
    try:
        yield
    finally:
        torch._C._get_graph_executor_optimize(old_optimize)