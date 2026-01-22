from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
Merges list of `EstimatorSpec` export_outputs for PREDICT.

    For each individual head, its DEFAULT_SERVING_KEY and PREDICT_SERVING_KEY
    are extracted and merged for `export_outputs` in PREDICT mode of
    `EstimatorSpec`. By default, the first head is served.

    Args:
      all_estimator_spec: list of `EstimatorSpec` for the individual heads.

    Returns:
      A dict of merged export_outputs from all heads for PREDICT.
    