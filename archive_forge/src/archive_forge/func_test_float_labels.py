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
def test_float_labels(self):
    """Asserts evaluation metrics for float labels in binary classification."""
    global_step = 100
    create_checkpoint((([[0.6, 0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0], [1.0]], [0.3])), global_step, self._model_dir)
    dnn_classifier = self._dnn_classifier_fn(hidden_units=(2, 2), feature_columns=[self._fc_impl.numeric_column('age')], model_dir=self._model_dir)

    def _input_fn():
        return ({'age': [[10.0], [10.0]]}, [[0.8], [0.4]])
    metrics = dnn_classifier.evaluate(input_fn=_input_fn, steps=1)
    self.assertAlmostEqual(2.731442, metrics[metric_keys.MetricKeys.LOSS])