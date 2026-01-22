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
def top_forward_activation_producers(self, top: int=10) -> List[LayerMemoryTrace]:
    """
        What are the top activation producers during the forward pass
        """
    return sorted(self.forward_traces, key=lambda a: a.event.memory_activations, reverse=True)[:top]