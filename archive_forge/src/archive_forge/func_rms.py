import tensorflow.compat.v2 as tf
from keras.src.optimizers import optimizer
from keras.src.saving.object_registration import register_keras_serializable
from tensorflow.python.util.tf_export import keras_export
def rms(x):
    return tf.sqrt(x + self.epsilon)