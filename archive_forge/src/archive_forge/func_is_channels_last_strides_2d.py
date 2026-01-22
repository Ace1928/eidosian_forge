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
def is_channels_last_strides_2d(self, sizes, strides) -> 'SymNode':
    return self._is_channels_last_strides_2d(sizes, strides)