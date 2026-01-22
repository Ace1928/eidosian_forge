import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
def with_stdout_output(self):
    """Print the result to stdout."""
    self._options['output'] = 'stdout'
    return self