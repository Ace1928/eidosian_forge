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
def predict_cluster_index(self, input_fn):
    """Finds the index of the closest cluster center to each input point.

    Args:
      input_fn: Input points. See `tf.estimator.Estimator.predict`.

    Yields:
      The index of the closest cluster center for each input point.
    """
    for index in self._predict_one_key(input_fn, KMeansClustering.CLUSTER_INDEX):
        yield index