from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
def update(self, runtime_secs, count):
    """Updates the unit time spent per iteration.

    Args:
      runtime_secs: The total elapsed time in seconds.
      count: The number of iterations.
    """
    if runtime_secs <= 0.0:
        tf.compat.v1.logging.debug('Invalid `runtime_secs`. Value must be positive. Actual:%.3f.', runtime_secs)
        return
    if count <= 0.0:
        tf.compat.v1.logging.debug('Invalid samples `count`. Value must be positive. Actual:%d.', count)
        return
    if len(self._buffer_wheel) >= self._capacity:
        self._buffer_wheel.popleft()
    step_time_secs = float(runtime_secs) / count
    self._buffer_wheel.append(RuntimeCounter(runtime_secs=runtime_secs, steps=count, step_time_secs=step_time_secs))
    self._sample_count += 1