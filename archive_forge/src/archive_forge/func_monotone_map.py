import dataclasses
import itertools
import sympy
from sympy.logic.boolalg import BooleanAtom, Boolean as SympyBoolean
import operator
import math
import logging
import torch
from typing import Union, Dict, Optional, SupportsFloat
from torch._prims_common import dtype_to_type
from .interp import sympy_interp
@classmethod
def monotone_map(cls, x, fn):
    """It's increasing or decreasing."""
    x = cls.wrap(x)
    l = fn(x.lower)
    u = fn(x.upper)
    return ValueRanges(min(l, u), max(l, u))