import enum
import platform
from absl import logging
from tensorflow.core.framework import dataset_options_pb2
from tensorflow.core.framework import model_pb2
from tensorflow.python.data.ops import test_mode
from tensorflow.python.data.util import options as options_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def merge(self, options):
    """Merges itself with the given `tf.data.Options`.

    If this object and the `options` to merge set an option differently, a
    warning is generated and this object's value is updated with the `options`
    object's value.

    Args:
      options: The `tf.data.Options` to merge with.

    Returns:
      New `tf.data.Options` object which is the result of merging self with
      the input `tf.data.Options`.
    """
    return options_lib.merge_options(self, options)