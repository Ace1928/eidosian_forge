import functools
from tensorflow.python.profiler.internal import _pywrap_traceme
from tensorflow.python.util.tf_export import tf_export
def trace_wrapper(trace_name, **trace_kwargs):
    """Decorator alternative to `with Trace(): ...`.  It's faster.

  Args:
    trace_name: The name of the trace event, or a callable to be traced, in
      which case the name is inferred from qualname or name of the callable.
    **trace_kwargs: Keyword arguments added to the trace event. Both the key and
      value are of types that can be converted to strings, which will be
      interpreted by the profiler according to the traceme name.

  Returns:
    A decorator that can wrap a function and apply `Trace` scope if needed,
    or a decorated function if used as a decorator directly.

  Example usage:
    ```python

    @trace_wrapper('trace_name')
    def func(x, y, z):
      pass  # code to execute and apply `Trace` if needed.

    # Equivalent to
    # with Trace('trace_name'):
    #   func(1, 2, 3)
    func(1, 2, 3)
    ```

  or
    ```python

    @trace_wrapper
    def func(x, y, z):
      pass  # code to execute and apply `Trace` if needed.

    # Equivalent to
    # with Trace(func.__qualname__):
    #   func(1, 2, 3)
    func(1, 2, 3)
    ```

  """
    if callable(trace_name):
        func = trace_name
        name = getattr(func, '__qualname__', None)
        if not name:
            name = getattr(func, '__name__', 'unknown function')
        return trace_wrapper(name)(func)

    def inner_wrapper(func):

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if enabled:
                with Trace(trace_name, **trace_kwargs):
                    return func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapped
    return inner_wrapper