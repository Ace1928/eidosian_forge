from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_FEATURE_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_LABEL_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_RECEIVER_DEFAULT_NAME
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
@estimator_export('estimator.export.build_parsing_serving_input_receiver_fn')
def build_parsing_serving_input_receiver_fn(feature_spec, default_batch_size=None):
    """Build a serving_input_receiver_fn expecting fed tf.Examples.

  Creates a serving_input_receiver_fn that expects a serialized tf.Example fed
  into a string placeholder.  The function parses the tf.Example according to
  the provided feature_spec, and returns all parsed Tensors as features.

  Args:
    feature_spec: a dict of string to `VarLenFeature`/`FixedLenFeature`.
    default_batch_size: the number of query examples expected per batch. Leave
      unset for variable batch size (recommended).

  Returns:
    A serving_input_receiver_fn suitable for use in serving.
  """

    def serving_input_receiver_fn():
        """An input_fn that expects a serialized tf.Example."""
        serialized_tf_example = tf.compat.v1.placeholder(dtype=tf.dtypes.string, shape=[default_batch_size], name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.compat.v1.io.parse_example(serialized_tf_example, feature_spec)
        return ServingInputReceiver(features, receiver_tensors)
    return serving_input_receiver_fn