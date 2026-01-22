import collections
import operator
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def stop_op_fn(var):
    placeholder = tf.compat.v1.placeholder_with_default(0, tuple(), name='stop_value')
    if self._stop_placeholder is None:
        self._stop_placeholder = placeholder
    return var.assign_add(placeholder)