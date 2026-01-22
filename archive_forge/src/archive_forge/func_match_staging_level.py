from tensorflow.python.autograph.operators import data_structures
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_util
def match_staging_level(value, like_value):
    """Casts a value to be staged at the same level as another."""
    if tensor_util.is_tf_type(like_value):
        return constant_op.constant(value)
    return value