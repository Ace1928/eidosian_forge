import uuid
import tensorflow.compat.v2 as tf
from tensorflow.python.eager.context import get_device_name
def is_cudnn_supported_inputs(mask, time_major, sequence_lengths):
    if tf.sysconfig.get_build_info()['is_rocm_build']:
        if not time_major and sequence_lengths is not None:
            return False
        if mask is not None:
            return tf.reduce_all(mask)
        elif sequence_lengths is not None:
            return tf.math.equal(tf.reduce_min(sequence_lengths), tf.reduce_max(sequence_lengths))
        else:
            return True
    if mask is None:
        return True
    if time_major:
        mask = tf.transpose(mask)
    return tf.logical_and(is_sequence_right_padded(mask), tf.logical_not(has_fully_masked_sequence(mask)))