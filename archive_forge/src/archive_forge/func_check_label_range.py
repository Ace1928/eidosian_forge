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
def check_label_range(labels, n_classes, message=None):
    """Check if labels are in the range of [0, n_classes)."""
    with ops.name_scope('check_label_range', values=(labels,)):
        if tf.executing_eagerly():
            assert_less = tf.reduce_all(tf.math.less_equal(labels, n_classes - 1))
            if not assert_less:
                raise ValueError(message or 'Labels must be <= {} - 1'.format(n_classes))
            assert_greater = tf.reduce_all(tf.math.greater_equal(labels, 0))
            if not assert_greater:
                raise ValueError(message or 'Labels must be >= 0')
            return labels
        assert_less = tf.compat.v1.debugging.assert_less_equal(labels, ops.convert_to_tensor(n_classes - 1, dtype=labels.dtype), message=message or 'Labels must be <= n_classes - 1')
        assert_greater = tf.compat.v1.debugging.assert_non_negative(labels, message=message or 'Labels must be >= 0')
        with tf.control_dependencies((assert_less, assert_greater)):
            return tf.identity(labels)