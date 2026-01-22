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
def test_one_dim_logits_with_batch_norm(self):
    """Tests one-dimensional logits.

    input_layer = [[10]]
    hidden_layer_0 = [[relu(0.6*10 +1), relu(0.5*10 -1)]] = [[7, 4]]
    hidden_layer_0 = [[relu(0.6*20 +1), relu(0.5*20 -1)]] = [[13, 9]]

    batch_norm_0, training (epsilon = 0.001):
      mean1 = 1/2*(7+13) = 10,
      variance1 = 1/2*(3^2+3^2) = 9
      x11 = (7-10)/sqrt(9+0.001) = -0.999944449,
      x21 = (13-10)/sqrt(9+0.001) = 0.999944449,

      mean2 = 1/2*(4+9) = 6.5,
      variance2 = 1/2*(2.5^2+.2.5^2) = 6.25
      x12 = (4-6.5)/sqrt(6.25+0.001) = -0.99992001,
      x22 = (9-6.5)/sqrt(6.25+0.001) = 0.99992001,

    logits = [[-1*(-0.999944449) + 2*(-0.99992001) + 0.3],
              [-1*0.999944449 + 2*0.99992001 + 0.3]]
           = [[-0.699895571],[1.299895571]]

    batch_norm_0, not training (epsilon = 0.001):
      moving_mean1 = 0, moving_variance1 = 1
      x11 = (7-0)/sqrt(1+0.001) = 6.996502623,
      x21 = (13-0)/sqrt(1+0.001) = 12.993504871,
      moving_mean2 = 0, moving_variance2 = 1
      x12 = (4-0)/sqrt(1+0.001) = 3.998001499,
      x22 = (9-0)/sqrt(1+0.001) = 8.995503372,

    logits = [[-1*6.996502623 + 2*3.998001499 + 0.3],
              [-1*12.993504871 + 2*8.995503372 + 0.3]]
           = [[1.299500375],[5.297501873]]
    """
    base_global_step = 100
    create_checkpoint((([[0.6, 0.5]], [1.0, -1.0]), ([[-1.0], [2.0]], [0.3])), base_global_step, self._model_dir, batch_norm_vars=([[0, 0], [1, 1], [0, 0], [1, 1]],))
    self._test_logits(ModeKeys.TRAIN, hidden_units=[2], logits_dimension=1, inputs=[[10.0], [20.0]], expected_logits=[[-0.699895571], [1.299895571]], batch_norm=True)
    for mode in [ModeKeys.EVAL, ModeKeys.PREDICT]:
        self._test_logits(mode, hidden_units=[2], logits_dimension=1, inputs=[[10.0], [20.0]], expected_logits=[[1.299500375], [5.297501873]], batch_norm=True)