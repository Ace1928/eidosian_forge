import contextlib
import dataclasses
import math
import textwrap
from typing import Any, Dict, Optional
import torch
from torch import inf
def tensor_totype(t):
    dtype = torch.float if t.is_mps else torch.double
    return t.to(dtype=dtype)