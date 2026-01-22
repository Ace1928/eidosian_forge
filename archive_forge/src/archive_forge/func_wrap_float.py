import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union
from torch import (  # noqa: F401
from torch.fx.experimental._sym_dispatch_mode import (
def wrap_float(self, num):
    assert type(num) is float
    import sympy
    return SymNode(sympy.Float(num), self.shape_env, float, num, constant=num, fx_node=num)