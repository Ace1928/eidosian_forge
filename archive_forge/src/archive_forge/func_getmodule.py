import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def getmodule(object):
    """TFDecorator-aware replacement for inspect.getmodule."""
    return _inspect.getmodule(object)