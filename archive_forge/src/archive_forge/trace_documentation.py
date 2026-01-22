import functools
from tensorflow.python.profiler.internal import _pywrap_traceme
from tensorflow.python.util.tf_export import tf_export
Sets metadata in this trace event.

    Args:
      **kwargs: metadata in key-value pairs.

    This method enables setting metadata in a trace event after it is
    created.

    Example usage:

    ```python

      def call(function):
        with tf.profiler.experimental.Trace("call",
             function_name=function.name) as tm:
          binary, in_cache = jit_compile(function)
          tm.set_metadata(in_cache=in_cache)
          execute(binary)

    ```
    In this example, we want to trace how much time spent on
    calling a function, which includes compilation and execution.
    The compilation can be either getting a cached copy of the
    binary or actually generating the binary, which is indicated
    by the boolean "in_cache" returned by jit_compile(). We need
    to use set_metadata() to pass in_cache because we did not know
    the in_cache value when the trace was created (and we cannot
    create the trace after jit_compile(), because we want
    to measure the entire duration of call()).
    