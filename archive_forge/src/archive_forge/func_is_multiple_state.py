import tensorflow.compat.v2 as tf
from keras.src.utils import control_flow_util
from tensorflow.python.platform import tf_logging as logging
def is_multiple_state(state_size):
    """Check whether the state_size contains multiple states."""
    return hasattr(state_size, '__len__') and (not isinstance(state_size, tf.TensorShape))