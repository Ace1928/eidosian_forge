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
def resolve_setitem(self, target, index, value):
    assert isinstance(index, types.Type), index
    fnty = self.resolve_value_type(operator.setitem)
    sig = fnty.get_call_type(self, (target, index, value), {})
    return sig