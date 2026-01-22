from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def create_checkpoint(weights_and_biases, global_step, model_dir, batch_norm_vars=None):
    """Create checkpoint file with provided model weights.

  Args:
    weights_and_biases: Iterable of tuples of weight and bias values.
    global_step: Initial global step to save in checkpoint.
    model_dir: Directory into which checkpoint is saved.
    batch_norm_vars: Variables used for batch normalization.
  """
    weights, biases = zip(*weights_and_biases)
    if batch_norm_vars:
        assert len(batch_norm_vars) == len(weights_and_biases) - 1
        bn_betas, bn_gammas, bn_means, bn_variances = zip(*batch_norm_vars)
    model_weights = {}
    for i in range(0, len(weights) - 1):
        model_weights[HIDDEN_WEIGHTS_NAME_PATTERN % i] = weights[i]
        model_weights[HIDDEN_BIASES_NAME_PATTERN % i] = biases[i]
        if batch_norm_vars:
            model_weights[BATCH_NORM_BETA_NAME_PATTERN % (i, i)] = bn_betas[i]
            model_weights[BATCH_NORM_GAMMA_NAME_PATTERN % (i, i)] = bn_gammas[i]
            model_weights[BATCH_NORM_MEAN_NAME_PATTERN % (i, i)] = bn_means[i]
            model_weights[BATCH_NORM_VARIANCE_NAME_PATTERN % (i, i)] = bn_variances[i]
    model_weights[LOGITS_WEIGHTS_NAME] = weights[-1]
    model_weights[LOGITS_BIASES_NAME] = biases[-1]
    with tf.Graph().as_default():
        for k, v in six.iteritems(model_weights):
            tf.Variable(v, name=k, dtype=tf.dtypes.float32)
        global_step_var = tf.compat.v1.train.create_global_step()
        with tf.compat.v1.Session() as sess:
            tf.compat.v1.initializers.global_variables().run()
            global_step_var.assign(global_step).eval()
            tf.compat.v1.train.Saver().save(sess, os.path.join(model_dir, 'model.ckpt'))