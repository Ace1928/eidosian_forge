from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union
import torch
import torio
from torch.utils._pytree import tree_map
@property
def num_out_streams(self):
    """Number of output streams configured by client code.

        :type: int
        """
    return self._be.num_out_streams()