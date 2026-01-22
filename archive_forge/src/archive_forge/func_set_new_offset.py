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
@staticmethod
def set_new_offset(relative_offset):
    torch.cuda._set_rng_state_offset(relative_offset.item())