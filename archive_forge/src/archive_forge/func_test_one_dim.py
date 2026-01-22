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
def test_one_dim(self):
    """Asserts train loss for one-dimensional input and logits."""
    base_global_step = 100
    hidden_units = (2, 2)
    create_checkpoint((([[0.6, 0.5]], [0.1, -0.1]), ([[1.0, 0.8], [-0.8, -1.0]], [0.2, -0.2]), ([[-1.0], [1.0]], [0.3])), base_global_step, self._model_dir)
    expected_loss = 9.4864
    opt = mock_optimizer(self, hidden_units=hidden_units, expected_loss=expected_loss)
    dnn_regressor = self._dnn_regressor_fn(hidden_units=hidden_units, feature_columns=(self._fc_impl.numeric_column('age'),), optimizer=opt, model_dir=self._model_dir)
    self.assertEqual(0, opt.minimize.call_count)
    num_steps = 5
    summary_hook = _SummaryHook()
    dnn_regressor.train(input_fn=lambda: ({'age': [[10.0]]}, [[1.0]]), steps=num_steps, hooks=(summary_hook,))
    self.assertEqual(1, opt.minimize.call_count)
    summaries = summary_hook.summaries()
    self.assertEqual(num_steps, len(summaries))
    for summary in summaries:
        _assert_simple_summary(self, {metric_keys.MetricKeys.LOSS_MEAN: expected_loss, 'dnn/dnn/hiddenlayer_0/fraction_of_zero_values': 0.0, 'dnn/dnn/hiddenlayer_1/fraction_of_zero_values': 0.5, 'dnn/dnn/logits/fraction_of_zero_values': 0.0, metric_keys.MetricKeys.LOSS: expected_loss}, summary)
    _assert_checkpoint(self, base_global_step + num_steps, input_units=1, hidden_units=hidden_units, output_units=1, model_dir=self._model_dir)