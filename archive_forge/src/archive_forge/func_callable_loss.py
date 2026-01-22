import tensorflow.compat.v2 as tf
from absl import app
from absl import flags
from keras.src import regularizers
from keras.src.testing_infra import test_utils
def callable_loss():
    return tf.reduce_sum(model.weights[0])