from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.tpu import util as util_lib
Constructs a RunConfig.

    Args:
      tpu_config: the TPUConfig that specifies TPU-specific configuration.
      evaluation_master: a string. The address of the master to use for eval.
        Defaults to master if not set.
      master: a string. The address of the master to use for training.
      cluster: a ClusterResolver
      **kwargs: keyword config parameters.

    Raises:
      ValueError: if cluster is not None and the provided session_config has a
        cluster_def already.

    @compatibility(TF2)
    TPU Estimator manages its own TensorFlow graph and session, so it is not
    compatible with TF2 behaviors. We recommend that you migrate to the newer
    `tf.distribute.TPUStrategy`. See the
    [TPU guide](https://www.tensorflow.org/guide/tpu) for details.
    @end_compatibility
    