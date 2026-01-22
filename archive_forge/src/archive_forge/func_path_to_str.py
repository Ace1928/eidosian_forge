from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import os
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def path_to_str(path):
    """Returns the file system path representation of a `PathLike` object, else as it is.

  Args:
    path: An object that can be converted to path representation.

  Returns:
    A `str` object.
  """
    if hasattr(path, '__fspath__'):
        path = tf.compat.as_str_any(path.__fspath__())
    return path