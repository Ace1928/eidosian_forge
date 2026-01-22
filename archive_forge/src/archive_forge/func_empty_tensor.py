import functools
import tensorflow as tf
def empty_tensor(shape, dtype):
    return tf.reshape(tf.convert_to_tensor((), dtype=dtype), shape=shape)