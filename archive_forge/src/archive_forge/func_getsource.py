import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def getsource(object):
    """TFDecorator-aware replacement for inspect.getsource."""
    return _inspect.getsource(tf_decorator.unwrap(object)[1])