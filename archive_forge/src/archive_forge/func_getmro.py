import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def getmro(cls):
    """TFDecorator-aware replacement for inspect.getmro."""
    return _inspect.getmro(cls)