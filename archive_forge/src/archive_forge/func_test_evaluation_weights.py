from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.inputs import pandas_io
def test_evaluation_weights(self):
    """Tests evaluation with weights."""
    with tf.Graph().as_default():
        tf.Variable([[11.0]], name=AGE_WEIGHT_NAME)
        tf.Variable([2.0], name=BIAS_NAME)
        tf.Variable(100, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
        save_variables_to_ckpt(self._model_dir)

    def _input_fn():
        features = {'age': ((1,), (1,)), 'weights': ((1.0,), (2.0,))}
        labels = ((10.0,), (10.0,))
        return (features, labels)
    linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('age'),), weight_column='weights', model_dir=self._model_dir)
    eval_metrics = linear_regressor.evaluate(input_fn=_input_fn, steps=1)
    self.assertDictEqual({metric_keys.MetricKeys.LOSS: 27.0, metric_keys.MetricKeys.LOSS_MEAN: 9.0, metric_keys.MetricKeys.PREDICTION_MEAN: 13.0, metric_keys.MetricKeys.LABEL_MEAN: 10.0, tf.compat.v1.GraphKeys.GLOBAL_STEP: 100}, eval_metrics)