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
def resolve_argument_type(self, val):
    """
        Return the numba type of a Python value that is being used
        as a function argument.  Integer types will all be considered
        int64, regardless of size.

        ValueError is raised for unsupported types.
        """
    try:
        return typeof(val, Purpose.argument)
    except ValueError:
        if numba.cuda.is_cuda_array(val):
            return typeof(numba.cuda.as_cuda_array(val, sync=False), Purpose.argument)
        else:
            raise