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
def update_metrics(self, eval_metrics, features, logits, labels, regularization_losses=None):
    """Updates eval metrics. See `base_head.Head` for details."""
    logits_dict = self._check_logits_and_labels(logits, labels)
    if regularization_losses is not None:
        regularization_loss = tf.math.add_n(regularization_losses)
        eval_metrics[self._loss_regularization_key].update_state(values=regularization_loss)
    for i, head in enumerate(self._heads):
        head_logits = logits_dict[head.name]
        head_labels = labels[head.name]
        training_loss = head.loss(logits=head_logits, labels=head_labels, features=features)
        eval_metrics[self._loss_keys[i]].update_state(values=training_loss)
        head_metrics = head.metrics()
        updated_metrics = head.update_metrics(head_metrics, features, head_logits, head_labels)
        eval_metrics.update(updated_metrics or {})
    return eval_metrics