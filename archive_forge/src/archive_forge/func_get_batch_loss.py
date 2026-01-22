from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
@abc.abstractmethod
def get_batch_loss(self, features, mode, state):
    """Return predictions, losses, and end state for a time series.

    Args:
      features: A dictionary with times, values, and (optionally) exogenous
        regressors. See `define_loss`.
      mode: The tf.estimator.ModeKeys mode to use (TRAIN, EVAL, INFER).
      state: Model-dependent state, each with size [batch size x ...]. The
        number and type will typically be fixed by the model (for example a mean
        and variance).

    Returns:
      A ModelOutputs object.
    """
    pass