import copy
from tensorflow.python.profiler import tfprof_logger
from tensorflow.python.util.tf_export import tf_export
def with_min_execution_time(self, min_micros=0, min_accelerator_micros=0, min_cpu_micros=0):
    """Only show profiler nodes consuming no less than 'min_micros'.

    Args:
      min_micros: Only show profiler nodes with execution time
          no less than this. It sums accelerator and cpu times.
      min_accelerator_micros: Only show profiler nodes spend no less than
          this time on accelerator (e.g. GPU).
      min_cpu_micros: Only show profiler nodes spend no less than
          this time on cpu.
    Returns:
      self
    """
    self._options['min_micros'] = min_micros
    self._options['min_accelerator_micros'] = min_accelerator_micros
    self._options['min_cpu_micros'] = min_cpu_micros
    return self