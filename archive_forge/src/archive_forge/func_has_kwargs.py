import functools
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def has_kwargs(fn):
    """Returns whether the passed callable has **kwargs in its signature.

  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).

  Returns:
    `bool`: if `fn` has **kwargs in its signature.

  Raises:
     `TypeError`: If fn is not a Function, or function-like object.
  """
    if isinstance(fn, functools.partial):
        fn = fn.func
    elif _is_callable_object(fn):
        fn = fn.__call__
    elif not callable(fn):
        raise TypeError(f'Argument `fn` should be a callable. Received: fn={fn} (of type {type(fn)})')
    return tf_inspect.getfullargspec(fn).varkw is not None