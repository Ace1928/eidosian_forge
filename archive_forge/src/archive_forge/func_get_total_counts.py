from collections import defaultdict
from typing import Any, Dict
import torch
from torch.utils._python_dispatch import TorchDispatchMode
def get_total_counts(self) -> int:
    return sum(self.comm_counts.values())