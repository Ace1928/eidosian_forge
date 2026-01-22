from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
Create a single vector from exogenous features.

    Args:
      times: A [batch size, window size] vector of times for this batch,
        primarily used to check the shape information of exogenous features.
      features: A dictionary of exogenous features corresponding to the columns
        in self._exogenous_feature_columns. Each value should have a shape
        prefixed by [batch size, window size].

    Returns:
      A Tensor with shape [batch size, window size, exogenous dimension], where
      the size of the exogenous dimension depends on the exogenous feature
      columns passed to the model's constructor.
    Raises:
      ValueError: If an exogenous feature has an unknown rank.
    