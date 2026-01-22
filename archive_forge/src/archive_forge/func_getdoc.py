import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def getdoc(object):
    """TFDecorator-aware replacement for inspect.getdoc.

  Args:
    object: An object, possibly decorated.

  Returns:
    The docstring associated with the object.

  The outermost-decorated object is intended to have the most complete
  documentation, so the decorated parameter is not unwrapped.
  """
    return _inspect.getdoc(object)