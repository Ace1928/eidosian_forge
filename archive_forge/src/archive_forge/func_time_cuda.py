from contextlib import contextmanager
from typing import Any, List, Tuple, cast
import random
import torch
import time
from torch.utils.benchmark import Timer
def time_cuda(fn, inputs, test_runs):
    t = Timer(stmt='fn(*inputs)', globals={'fn': fn, 'inputs': inputs})
    times = t.blocked_autorange()
    return times.median * 1000