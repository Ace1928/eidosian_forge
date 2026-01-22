import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic
def get_payload_struct(context, builder, set_type, ptr):
    """
    Given a set value and type, get its payload structure (as a
    reference, so that mutations are seen by all).
    """
    payload_type = types.SetPayload(set_type)
    ptrty = context.get_data_type(payload_type).as_pointer()
    payload = builder.bitcast(ptr, ptrty)
    return context.make_data_helper(builder, payload_type, ref=payload)