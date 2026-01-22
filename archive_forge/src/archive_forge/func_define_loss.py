from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def define_loss(self, features, mode):
    """Default loss definition with state replicated across a batch.

    Time series passed to this model have a batch dimension, and each series in
    a batch can be operated on in parallel. This loss definition assumes that
    each element of the batch represents an independent sample conditioned on
    the same initial state (i.e. it is simply replicated across the batch). A
    batch size of one provides sequential operations on a single time series.

    More complex processing may operate instead on get_start_state() and
    get_batch_loss() directly.

    Args:
      features: A dictionary (such as is produced by a chunker) with at minimum
        the following key/value pairs (others corresponding to the
        `exogenous_feature_columns` argument to `__init__` may be included
        representing exogenous regressors):
        TrainEvalFeatures.TIMES: A [batch size x window size] integer Tensor
          with times for each observation. If there is no artificial chunking,
          the window size is simply the length of the time series.
        TrainEvalFeatures.VALUES: A [batch size x window size x num features]
          Tensor with values for each observation.
      mode: The tf.estimator.ModeKeys mode to use (TRAIN, EVAL). For INFER, see
        predict().

    Returns:
      A ModelOutputs object.
    """
    self._check_graph_initialized()
    start_state = math_utils.replicate_state(start_state=self.get_start_state(), batch_size=tf.compat.v1.shape(features[TrainEvalFeatures.TIMES])[0])
    return self.get_batch_loss(features=features, mode=mode, state=start_state)