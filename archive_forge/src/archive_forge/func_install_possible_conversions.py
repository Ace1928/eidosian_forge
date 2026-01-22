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
def install_possible_conversions(self, actualargs, formalargs):
    """
        Install possible conversions from the actual argument types to
        the formal argument types in the C++ type manager.
        Return True if all arguments can be converted.
        """
    if len(actualargs) != len(formalargs):
        return False
    for actual, formal in zip(actualargs, formalargs):
        if self.tm.check_compatible(actual, formal) is not None:
            continue
        conv = self.can_convert(actual, formal)
        if conv is None:
            return False
        assert conv is not Conversion.exact
        self.tm.set_compatible(actual, formal, conv)
    return True