from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.feature_column.feature_column import _NumericColumn
from tensorflow.python.framework import ops
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
def update_metric_with_broadcast_weights(eval_metric, values, weights):
    values = tf.cast(values, dtype=tf.dtypes.float32)
    if weights is not None:
        weights = tf.compat.v2.__internal__.ops.broadcast_weights(weights, values)
    eval_metric.update_state(values=values, sample_weight=weights)