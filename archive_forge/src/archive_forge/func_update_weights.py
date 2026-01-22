from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
from six.moves import range
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import internal_convert_to_tensor
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.ops import gen_sdca_ops
from tensorflow.python.ops import variables as var_ops
from tensorflow.python.ops.nn import log_poisson_loss
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils.sharded_mutable_dense_hashtable import _ShardedMutableDenseHashTable
def update_weights(self, train_op):
    """Updates the model weights.

    This function must be called on at least one worker after `minimize`.
    In distributed training this call can be omitted on non-chief workers to
    speed up training.

    Args:
      train_op: The operation returned by the `minimize` call.

    Returns:
      An Operation that updates the model weights.
    """
    with tf.control_dependencies([train_op]):
        update_ops = []
        for name in ['sparse_features_weights', 'dense_features_weights']:
            for var, slot_var in zip(self._variables[name], self._slots['unshrunk_' + name]):
                for v, sv in zip(self._var_to_list(var), self._var_to_list(slot_var)):
                    update_ops.append(v.assign(sv))
    if self._symmetric_l1_regularization() > 0:
        shrinkage = self._symmetric_l1_regularization() / self._symmetric_l2_regularization()
        with tf.control_dependencies(update_ops):
            update_ops = []
            for name in ['sparse_features_weights', 'dense_features_weights']:
                for var in self._variables[name]:
                    for v in self._var_to_list(var):
                        with tf.compat.v1.device(v.device):
                            v_shrunk = tf.math.sign(v) * tf.math.maximum(0.0, tf.math.abs(v) - shrinkage)
                            update_ops.append(v.assign(v_shrunk))
            return tf.group(*update_ops)
    else:
        return tf.group(*update_ops)