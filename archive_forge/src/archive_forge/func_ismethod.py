import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def ismethod(object):
    """TFDecorator-aware replacement for inspect.ismethod."""
    return _inspect.ismethod(tf_decorator.unwrap(object)[1])