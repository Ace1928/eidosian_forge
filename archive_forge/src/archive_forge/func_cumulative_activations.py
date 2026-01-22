from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from fairscale.nn import FullyShardedDataParallel
def cumulative_activations(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
    event_allocations = [t.event.memory_activations for t in memory_traces]
    x, y_forward, y_backward = self._split_forward_backward(memory_traces, event_allocations)
    cumulative_forward_activations = np.cumsum(y_forward)
    ax.plot(x, cumulative_forward_activations, label=job_name)
    self._y_axis_in_gigabytes(ax)