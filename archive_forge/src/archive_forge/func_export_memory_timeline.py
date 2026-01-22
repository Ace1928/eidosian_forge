import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
def export_memory_timeline(self, path, device) -> None:
    """Saves the memory timeline as [times, sizes by category]
        as a JSON formatted file to the given path for the given
        device."""
    times, sizes = self._coalesce_timeline(device)
    import json
    with open(path, 'w') as f:
        json.dump([times, sizes], f)