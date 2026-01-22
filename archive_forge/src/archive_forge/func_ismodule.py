import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def ismodule(object):
    """TFDecorator-aware replacement for inspect.ismodule."""
    return _inspect.ismodule(tf_decorator.unwrap(object)[1])