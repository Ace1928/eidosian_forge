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
def test_multi_feature_column_mix_multi_dim_logits(self):
    """Tests multiple feature columns and multi-dimensional logits.

    All numbers are the same as test_multi_dim_input_multi_dim_logits. The only
    difference is that the input consists of two 1D feature columns, instead of
    one 2D feature column.
    """
    base_global_step = 100
    create_checkpoint((([[0.6, 0.5], [-0.6, -0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]], [0.3, -0.3, 0.0])), base_global_step, self._model_dir)
    hidden_units = (2, 2)
    logits_dimension = 3
    inputs = ([[10.0]], [[8.0]])
    expected_logits = [[-0.48, 0.48, 0.39]]
    for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
        with tf.Graph().as_default():
            tf.compat.v1.train.create_global_step()
            with tf.compat.v1.variable_scope('dnn'):
                input_layer_partitioner = tf.compat.v1.min_max_variable_partitioner(max_partitions=0, min_slice_size=64 << 20)
                logit_fn = self._dnn_logit_fn_builder(units=logits_dimension, hidden_units=hidden_units, feature_columns=[feature_column.numeric_column('age'), tf.feature_column.numeric_column('height')], activation_fn=tf.nn.relu, dropout=None, input_layer_partitioner=input_layer_partitioner, batch_norm=False)
                logits = logit_fn(features={'age': tf.constant(inputs[0]), 'height': tf.constant(inputs[1])}, mode=mode)
                with tf.compat.v1.train.MonitoredTrainingSession(checkpoint_dir=self._model_dir) as sess:
                    self.assertAllClose(expected_logits, sess.run(logits))