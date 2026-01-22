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
def check_logits_final_dim(logits, expected_logits_dimension):
    """Checks that logits shape is [D0, D1, ... DN, logits_dimension]."""
    with ops.name_scope('logits', values=(logits,)) as scope:
        logits = tf.cast(logits, tf.dtypes.float32)
        if tf.executing_eagerly():
            logits_shape = logits._shape_tuple()
            logits_rank = logits._rank()
            if logits_rank < 2:
                raise ValueError('logits must have rank at least 2.  Received rank {}, shape {}'.format(logits_rank, logits_shape))
            if isinstance(expected_logits_dimension, int) and logits_shape[-1] != expected_logits_dimension:
                raise ValueError('logits shape must be [D0, D1, ... DN, logits_dimension], got {}.'.format(logits_shape))
            return logits
        logits_shape = tf.compat.v1.shape(logits)
        assert_rank = tf.compat.v1.debugging.assert_rank_at_least(logits, 2, data=[logits_shape], message='logits shape must be [D0, D1, ... DN, logits_dimension]')
        with tf.control_dependencies([assert_rank]):
            static_shape = logits.shape
            if static_shape.ndims is not None and static_shape[-1] is not None:
                if isinstance(expected_logits_dimension, int) and static_shape[-1] != expected_logits_dimension:
                    raise ValueError('logits shape must be [D0, D1, ... DN, logits_dimension], got {}.'.format(static_shape))
                return logits
            assert_dimension = tf.compat.v1.debugging.assert_equal(expected_logits_dimension, logits_shape[-1], data=[logits_shape], message='logits shape must be [D0, D1, ... DN, logits_dimension]')
            with tf.control_dependencies([assert_dimension]):
                return tf.identity(logits, name=scope)