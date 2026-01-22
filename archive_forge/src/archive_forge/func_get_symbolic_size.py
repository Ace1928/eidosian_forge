from __future__ import annotations
import collections
import dataclasses
import itertools
import pprint
from typing import Any, Dict, Iterable, List, Optional, Protocol
import sympy
import torch
from .. import config, ir
from ..utils import cache_on_self, CachedMethod, IndentedBuffer
from ..virtualized import V
from .wrapper import (
@cache_on_self
def get_symbolic_size(self) -> sympy.Expr:
    return align(self.left.get_symbolic_size()) + self.right.get_symbolic_size()