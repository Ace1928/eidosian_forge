import tensorflow.compat.v2 as tf
from keras.src.utils import control_flow_util
from tensorflow.python.platform import tf_logging as logging
def generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        batch_size = tf.shape(inputs)[0]
        dtype = inputs.dtype
    return generate_zero_filled_state(batch_size, cell.state_size, dtype)