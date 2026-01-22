from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clustering_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
def transform(self, input_fn):
    """Transforms each input point to its distances to all cluster centers.

    Note that if `distance_metric=KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE`,
    this
    function returns the squared Euclidean distance while the corresponding
    sklearn function returns the Euclidean distance.

    Args:
      input_fn: Input points. See `tf.estimator.Estimator.predict`.

    Yields:
      The distances from each input point to each cluster center.
    """
    for distances in self._predict_one_key(input_fn, KMeansClustering.ALL_DISTANCES):
        if self._distance_metric == KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE:
            yield np.sqrt(distances)
        else:
            yield distances