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
def call_loss_fn(loss_fn, labels, logits, features, expected_loss_dim=1):
    """Calls loss_fn and checks the returned shape.

  For shape checking, eager uses the static dimension to improve performance.

  Args:
    loss_fn: The loss function.
    labels: Processed labels Tensor.
    logits: Logits Tensor of shape [D0, D1, ... DN, logits_dimension].
    features: Features dict.
    expected_loss_dim: The expected last dimension of loss Tensor.

  Returns:
    Loss Tensor with shape [D0, D1, ... DN, expected_loss_dim].

  Raises:
    ValueError: If the loss tensor shape is unexpected.
  """
    loss_fn_args = function_utils.fn_args(loss_fn)
    kwargs = {}
    if 'features' in loss_fn_args:
        kwargs['features'] = features
    with ops.name_scope('call_loss_fn', values=[labels, logits] + list(six.itervalues(features))):
        unweighted_loss = loss_fn(labels=labels, logits=logits, **kwargs)
        if tf.executing_eagerly():
            loss_shape = unweighted_loss._shape_tuple()
            logits_shape = logits._shape_tuple()
            expected_loss_shape = logits_shape[:-1] + (expected_loss_dim,)
            if loss_shape != expected_loss_shape:
                raise ValueError('loss_fn must return Tensor of shape [D0, D1, ... DN, {}]. '.format(expected_loss_dim), 'logits_shape: ', logits_shape, 'loss_shape: ', loss_shape)
            return unweighted_loss
        logits_shape = tf.compat.v1.shape(logits, name='logits_shape')
        expected_loss_shape = tf.concat([logits_shape[:-1], [expected_loss_dim]], axis=0, name='expected_loss_shape')
        loss_shape = tf.compat.v1.shape(unweighted_loss, name='loss_shape')
        check_loss_shape_op = tf.debugging.Assert(tf.reduce_all(tf.math.equal(loss_shape, expected_loss_shape)), data=['loss_fn must return Tensor of shape [D0, D1, ... DN, {}]. '.format(expected_loss_dim), 'logits_shape: ', logits_shape, 'loss_shape: ', loss_shape], name='check_loss_shape')
        with tf.control_dependencies([check_loss_shape_op]):
            return tf.identity(unweighted_loss)