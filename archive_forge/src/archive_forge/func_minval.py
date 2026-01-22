import enum
import numpy as np
from .abstract import Dummy, Hashable, Literal, Number, Type
from functools import total_ordering, cached_property
from numba.core import utils
from numba.core.typeconv import Conversion
from numba.np import npdatetime_helpers
@property
def minval(self):
    """
        The minimal value representable by this type.
        """
    if self.signed:
        return -(1 << self.bitwidth - 1)
    else:
        return 0