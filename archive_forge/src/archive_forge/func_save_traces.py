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
def save_traces(self, path: str) -> None:
    """
        Save the traces in a JSON file
        """
    import json
    with open(path, 'w') as f:
        json_traces = [t.to_dict() for t in self.memory_traces]
        json.dump({'traces': json_traces}, f)