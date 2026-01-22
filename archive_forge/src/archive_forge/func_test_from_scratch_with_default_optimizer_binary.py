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
def test_from_scratch_with_default_optimizer_binary(self):
    hidden_units = (2, 2)
    dnn_classifier = self._dnn_classifier_fn(hidden_units=hidden_units, feature_columns=(self._fc_impl.numeric_column('age'),), model_dir=self._model_dir)
    num_steps = 5
    dnn_classifier.train(input_fn=lambda: ({'age': [[10.0]]}, [[1]]), steps=num_steps)
    _assert_checkpoint(self, num_steps, input_units=1, hidden_units=hidden_units, output_units=1, model_dir=self._model_dir)