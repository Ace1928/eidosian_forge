import builtins
import inspect
import itertools
import linecache
import sys
import threading
import types
from tensorflow.python.util import tf_inspect
def islambda(f):
    if not tf_inspect.isfunction(f):
        return False
    if not (hasattr(f, '__name__') and hasattr(f, '__code__')):
        return False
    return f.__name__ == '<lambda>' or f.__code__.co_name == '<lambda>'