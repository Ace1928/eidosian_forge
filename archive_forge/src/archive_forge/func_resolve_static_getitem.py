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
def resolve_static_getitem(self, value, index):
    assert not isinstance(index, types.Type), index
    args = (value, index)
    kws = ()
    return self.resolve_function_type('static_getitem', args, kws)