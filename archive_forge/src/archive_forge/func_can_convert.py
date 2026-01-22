from collections import defaultdict
from collections.abc import Sequence
import types as pytypes
import weakref
import threading
import contextlib
import operator
import numba
from numba.core import types, errors
from numba.core.typeconv import Conversion, rules
from numba.core.typing import templates
from numba.core.utils import order_by_target_specificity
from .typeof import typeof, Purpose
from numba.core import utils
def can_convert(self, fromty, toty):
    """
        Check whether conversion is possible from *fromty* to *toty*.
        If successful, return a numba.typeconv.Conversion instance;
        otherwise None is returned.
        """
    if fromty == toty:
        return Conversion.exact
    else:
        conv = self.tm.check_compatible(fromty, toty)
        if conv is not None:
            return conv
        forward = fromty.can_convert_to(self, toty)
        backward = toty.can_convert_from(self, fromty)
        if backward is None:
            return forward
        elif forward is None:
            return backward
        else:
            return min(forward, backward)