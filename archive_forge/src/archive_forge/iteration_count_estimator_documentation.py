from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import tensorflow as tf
Gets the iterations count estimate.

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
    