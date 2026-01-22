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
def expect_true(self, file, line):
    if self.has_hint():
        return self.guard_bool(file, line)
    return self.shape_env.defer_runtime_assert(self.expr, f'{file}:{line}', fx_node=self.fx_node)