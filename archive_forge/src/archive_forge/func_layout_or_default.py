from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def layout_or_default(layout: Optional[torch.layout]) -> torch.layout:
    return layout if layout is not None else torch.strided