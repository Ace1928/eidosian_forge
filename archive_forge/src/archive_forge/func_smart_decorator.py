import os
from io import open
import types
from functools import wraps, partial
from contextlib import contextmanager
import sys, re
import sre_parse
import sre_constants
from inspect import getmembers, getmro
from functools import partial, wraps
from itertools import repeat, product
def smart_decorator(f, create_decorator):
    if isinstance(f, types.FunctionType):
        return wraps(f)(create_decorator(f, True))
    elif isinstance(f, (classtype, type, types.BuiltinFunctionType)):
        return wraps(f)(create_decorator(f, False))
    elif isinstance(f, types.MethodType):
        return wraps(f)(create_decorator(f.__func__, True))
    elif isinstance(f, partial):
        return wraps(f.func)(create_decorator(lambda *args, **kw: f(*args[1:], **kw), True))
    else:
        return create_decorator(f.__func__.__call__, True)