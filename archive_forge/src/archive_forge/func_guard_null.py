import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
def guard_null(context, builder, value, exc_tuple):
    """
    Guard against *value* being null or zero.
    *exc_tuple* should be a (exception type, arguments...) tuple.
    """
    with builder.if_then(is_scalar_zero(builder, value), likely=False):
        exc = exc_tuple[0]
        exc_args = exc_tuple[1:] or None
        context.call_conv.return_user_exc(builder, exc, exc_args)