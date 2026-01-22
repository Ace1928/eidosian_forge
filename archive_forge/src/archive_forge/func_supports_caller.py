import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def supports_caller(func):
    """Apply a caller_stack compatibility decorator to a plain
    Python function.

    See the example in :ref:`namespaces_python_modules`.

    """

    def wrap_stackframe(context, *args, **kwargs):
        context.caller_stack._push_frame()
        try:
            return func(context, *args, **kwargs)
        finally:
            context.caller_stack._pop_frame()
    return wrap_stackframe