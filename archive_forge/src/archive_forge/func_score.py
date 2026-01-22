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
def score(self, input_fn):
    """Returns the sum of squared distances to nearest clusters.

    Note that this function is different from the corresponding one in sklearn
    which returns the negative sum.

    Args:
      input_fn: Input points. See `tf.estimator.Estimator.evaluate`. Only one
        batch is retrieved.

    Returns:
      The sum of the squared distance from each point in the first batch of
      inputs to its nearest cluster center.
    """
    return self.evaluate(input_fn=input_fn, steps=1)[KMeansClustering.SCORE]