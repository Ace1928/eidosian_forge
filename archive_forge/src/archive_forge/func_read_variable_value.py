import uuid
import tensorflow.compat.v2 as tf
from tensorflow.python.eager.context import get_device_name
def read_variable_value(v):
    """Read the value of a variable if it is variable."""
    if isinstance(v, tf.Variable):
        return v.read_value()
    return v