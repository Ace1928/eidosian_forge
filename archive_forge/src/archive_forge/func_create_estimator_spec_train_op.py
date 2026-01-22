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
def create_estimator_spec_train_op(head_name, optimizer=None, trainable_variables=None, train_op_fn=None, update_ops=None, regularized_training_loss=None, loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE):
    """Create train_op for estimator_spec.

  Args:
    head_name: The name of the head.
    optimizer: An `tf.keras.optimizers.Optimizer` instance to optimize the loss
      in TRAIN mode. Namely, sets `train_op = optimizer.get_updates(loss,
      trainable_variables)`, which updates variables to minimize `loss`.
    trainable_variables: A list or tuple of `Variable` objects to update to
      minimize `loss`. In Tensorflow 1.x, by default these are the list of
      variables collected in the graph under the key
      `GraphKeys.TRAINABLE_VARIABLES`. As Tensorflow 2.x doesn't have
      collections and GraphKeys, trainable_variables need to be passed
      explicitly here.
    train_op_fn: Function that takes a scalar loss `Tensor` and returns
      `train_op`. Used if `optimizer` is `None`.
    update_ops: A list or tuple of update ops to be run at training time. For
      example, layers such as BatchNormalization create mean and variance update
      ops that need to be run at training time. In Tensorflow 1.x, these are
      thrown into an UPDATE_OPS collection. As Tensorflow 2.x doesn't have
      collections, update_ops need to be passed explicitly here.
    regularized_training_loss: A scalar for total training loss that includes
      all regularization losses. If you're not using optimizer to generate train
      op, make sure to scale the loss correctly before passing it in. The loss
      typically needs to be scaled down by the number of workers.
    loss_reduction: One of `tf.keras.losses.Reduction` except `NONE`. Describes
      how to reduce training loss over batch. Defaults to `SUM_OVER_BATCH_SIZE`.

  Returns:
    A train op for EstimatorSpec.
  """
    del head_name
    validate_update_ops(update_ops)
    with ops.name_scope(''):
        with ops.name_scope('training'):
            if optimizer is not None:
                if train_op_fn is not None:
                    raise ValueError('train_op_fn and optimizer cannot both be set.')
                validate_v2_optimizer(optimizer)
                validate_trainable_variables(trainable_variables)
                if loss_reduction == tf.losses.Reduction.SUM_OVER_BATCH_SIZE:
                    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
                    if num_replicas > 1:
                        regularized_training_loss *= 1.0 / num_replicas
                train_op = optimizer.get_updates(regularized_training_loss, trainable_variables)[0]
            elif train_op_fn is not None:
                train_op = train_op_fn(regularized_training_loss)
            else:
                raise ValueError('train_op_fn and optimizer cannot both be None.')
            if update_ops is not None:
                train_op = tf.group(train_op, *update_ops)
            return train_op