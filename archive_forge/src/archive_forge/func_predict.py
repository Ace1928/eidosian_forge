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
def predict(self, features):
    """Returns predictions of future observations given an initial state.

    Computes distributions for future observations. For sampled draws from the
    model where each is conditioned on the previous, see generate().

    Args:
      features: A dictionary with at minimum the following key/value pairs
        (others corresponding to the `exogenous_feature_columns` argument to
        `__init__` may be included representing exogenous regressors):
        PredictionFeatures.TIMES: A [batch size x window size] Tensor with times
          to make predictions for. Times must be increasing within each part of
          the batch, and must be greater than the last time `state` was updated.
        PredictionFeatures.STATE_TUPLE: Model-dependent state, each with size
          [batch size x ...]. The number and type will typically be fixed by the
          model (for example a mean and variance). Typically these will be the
          end state returned by get_batch_loss, predicting beyond that data.

    Returns:
      A dictionary with model-dependent predictions corresponding to the
      requested times. Keys indicate the type of prediction, and values have
      shape [batch size x window size x ...]. For example state space models
      return a "predicted_mean" and "predicted_covariance".
    """
    pass