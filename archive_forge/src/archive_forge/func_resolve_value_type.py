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
def resolve_value_type(self, val):
    """
        Return the numba type of a Python value that is being used
        as a runtime constant.
        ValueError is raised for unsupported types.
        """
    try:
        ty = typeof(val, Purpose.constant)
    except ValueError as e:
        typeof_exc = utils.erase_traceback(e)
    else:
        return ty
    if isinstance(val, types.ExternalFunction):
        return val
    ty = self._get_global_type(val)
    if ty is not None:
        return ty
    raise typeof_exc