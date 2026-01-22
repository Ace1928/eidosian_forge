import functools
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def get_func_name(func):
    """Returns name of passed callable."""
    _, func = tf_decorator.unwrap(func)
    if callable(func):
        if tf_inspect.isfunction(func):
            return func.__name__
        elif tf_inspect.ismethod(func):
            return '%s.%s' % (six.get_method_self(func).__class__.__name__, six.get_method_function(func).__name__)
        else:
            return str(type(func))
    else:
        raise ValueError(f'Argument `func` must be a callable. Received func={func} (of type {type(func)})')