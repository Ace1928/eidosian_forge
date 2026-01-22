from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
def is_buffer_ready(self) -> bool:
    """Returns true if all the output streams have at least one chunk filled."""
    return self._be.is_buffer_ready()