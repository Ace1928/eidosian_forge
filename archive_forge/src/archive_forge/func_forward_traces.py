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
@property
def forward_traces(self) -> List[LayerMemoryTrace]:
    """
        Get the part of the traces which corresponds to the forward pass
        """
    return [t for t in self.memory_traces if t.is_forward]