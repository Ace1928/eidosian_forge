import math
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_replication
def get_sequence_length_feature_key_name_from_feature_key_name(feature_name):
    """Gets the name of the sequence length feature from that of the base feature.

  Args:
    feature_name: The feature key of a sequence column.

  Returns:
    A string which is the feature key for the associated feature length column.
  """
    return feature_name + _SEQUENCE_FEATURE_LENGTH_POSTFIX