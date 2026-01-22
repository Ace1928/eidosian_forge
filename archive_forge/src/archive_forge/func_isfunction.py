import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def isfunction(object):
    """TFDecorator-aware replacement for inspect.isfunction."""
    return _inspect.isfunction(tf_decorator.unwrap(object)[1])