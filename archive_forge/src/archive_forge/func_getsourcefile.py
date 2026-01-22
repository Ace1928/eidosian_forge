import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def getsourcefile(object):
    """TFDecorator-aware replacement for inspect.getsourcefile."""
    return _inspect.getsourcefile(tf_decorator.unwrap(object)[1])