from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def loss(self, labels, logits, features=None, mode=None, regularization_losses=None):
    """Returns regularized training loss. See `base_head.Head` for details."""
    logits_dict = self._check_logits_and_labels(logits, labels)
    training_losses = []
    for head in self._heads:
        training_loss = head.loss(logits=logits_dict[head.name], labels=labels[head.name], features=features, mode=mode)
        training_losses.append(training_loss)
    training_losses = tuple(training_losses)
    with ops.name_scope('merge_losses', values=training_losses + (self._head_weights or tuple())):
        if self._head_weights:
            head_weighted_training_losses = []
            for training_loss, head_weight in zip(training_losses, self._head_weights):
                head_weighted_training_losses.append(tf.math.multiply(training_loss, head_weight))
            training_losses = head_weighted_training_losses
        merged_training_loss = tf.math.add_n(training_losses)
        regularization_loss = tf.math.add_n(regularization_losses) if regularization_losses is not None else None
        regularized_training_loss = merged_training_loss + regularization_loss if regularization_loss is not None else merged_training_loss
    return regularized_training_loss