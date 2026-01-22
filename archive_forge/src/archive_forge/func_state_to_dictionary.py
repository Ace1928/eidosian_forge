from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.export import export_lib
def state_to_dictionary(state_tuple):
    """Flatten model state into a dictionary with string keys."""
    flattened = {}
    for state_number, state_value in enumerate(tf.nest.flatten(state_tuple)):
        prefixed_state_name = '{}_{:02d}'.format(feature_keys.State.STATE_PREFIX, state_number)
        flattened[prefixed_state_name] = state_value
    return flattened