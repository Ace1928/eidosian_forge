import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def isframe(object):
    """TFDecorator-aware replacement for inspect.ismodule."""
    return _inspect.isframe(tf_decorator.unwrap(object)[1])