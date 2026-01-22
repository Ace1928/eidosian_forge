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
def test_evaluation_for_multi_dimensions(self):
    x_dim = 3
    label_dim = 2
    with tf.Graph().as_default():
        tf.Variable([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name=AGE_WEIGHT_NAME)
        tf.Variable([7.0, 8.0], name=BIAS_NAME)
        tf.Variable(100, name='global_step', dtype=tf.dtypes.int64)
        save_variables_to_ckpt(self._model_dir)
    linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('age', shape=(x_dim,)),), label_dimension=label_dim, model_dir=self._model_dir)
    input_fn = numpy_io.numpy_input_fn(x={'age': np.array([[2.0, 4.0, 5.0]])}, y=np.array([[46.0, 58.0]]), batch_size=1, num_epochs=None, shuffle=False)
    eval_metrics = linear_regressor.evaluate(input_fn=input_fn, steps=1)
    self.assertItemsEqual((metric_keys.MetricKeys.LOSS, metric_keys.MetricKeys.LOSS_MEAN, metric_keys.MetricKeys.PREDICTION_MEAN, metric_keys.MetricKeys.LABEL_MEAN, tf.compat.v1.GraphKeys.GLOBAL_STEP), eval_metrics.keys())
    self.assertAlmostEqual(0, eval_metrics[metric_keys.MetricKeys.LOSS])