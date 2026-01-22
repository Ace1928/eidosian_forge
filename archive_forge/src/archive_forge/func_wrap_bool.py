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
def wrap_bool(self, num):
    assert type(num) is bool
    import sympy
    return SymNode(sympy.true if num else sympy.false, self.shape_env, bool, num, constant=num, fx_node=num)