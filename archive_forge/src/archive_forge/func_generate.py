from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def generate(self, number_of_series, series_length, model_parameters=None, seed=None):
    """Sample synthetic data from model parameters, with optional substitutions.

    Returns `number_of_series` possible sequences of future values, sampled from
    the generative model with each conditioned on the previous. Samples are
    based on trained parameters, except for those parameters explicitly
    overridden in `model_parameters`.

    For distributions over future observations, see predict().

    Args:
      number_of_series: Number of time series to create.
      series_length: Length of each time series.
      model_parameters: A dictionary mapping model parameters to values, which
        replace trained parameters when generating data.
      seed: If specified, return deterministic time series according to this
        value.

    Returns:
      A dictionary with keys TrainEvalFeatures.TIMES (mapping to an array with
      shape [number_of_series, series_length]) and TrainEvalFeatures.VALUES
      (mapping to an array with shape [number_of_series, series_length,
      num_features]).
    """
    raise NotImplementedError('This model does not support generation.')