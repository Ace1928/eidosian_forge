from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
def get(self, total_secs):
    """Gets the iterations count estimate.

    If recent predicted iterations are stable, re-use the previous value.
    Otherwise, update the prediction value based on the delta between the
    current prediction and the expected number of iterations as determined by
    the per-step runtime.

    Args:
      total_secs: The target runtime in seconds.

    Returns:
      The number of iterations as estimate.

    Raise:
      ValueError: If `total_secs` value is not positive.
    """
    if total_secs <= 0:
        raise ValueError('Invalid `total_secs`. It must be positive number. Actual:%d' % total_secs)
    if not self._buffer_wheel:
        tf.compat.v1.logging.debug('IterationCountEstimator has no sample(s). Returns min iterations:%d.', self._min_iterations)
        return self._min_iterations
    mean_runtime_secs = self._mean_runtime_secs()
    mean_step_time_secs = self._mean_step_time_secs()
    std_step_time_secs = self._std_step_time_secs()
    projected_iterations = total_secs / mean_step_time_secs
    last_runtime_secs = self._buffer_wheel[-1].runtime_secs
    delta_iterations = projected_iterations - self._last_iterations
    if (self._diff_less_than_percentage(last_runtime_secs, total_secs, 10) or self._diff_less_than_percentage(mean_runtime_secs, total_secs, 5)) and self._is_step_time_stable():
        delta_iterations = 0
    self._last_iterations += delta_iterations
    self._last_iterations = max(self._last_iterations, self._min_iterations)
    tf.compat.v1.logging.info('IterationCountEstimator -- target_runtime:%.3fs. last_runtime:%.3fs. mean_runtime:%.3fs. last_step_time:%.3f. std_step_time:%.3f. mean_step_time:%.3fs. delta_steps:%.2f. prev_steps:%.2f. next_steps:%.2f.', total_secs, last_runtime_secs, mean_runtime_secs, self._buffer_wheel[-1].step_time_secs, std_step_time_secs, mean_step_time_secs, delta_iterations, self._buffer_wheel[-1].steps, self._last_iterations)
    return int(self._last_iterations + 0.5)