import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def isroutine(object):
    """TFDecorator-aware replacement for inspect.isroutine."""
    return _inspect.isroutine(tf_decorator.unwrap(object)[1])