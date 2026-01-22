import functools
from tensorflow.python.profiler.internal import _pywrap_traceme
from tensorflow.python.util.tf_export import tf_export
def inner_wrapper(func):

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if enabled:
            with Trace(trace_name, **trace_kwargs):
                return func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapped