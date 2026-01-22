import types
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src.metrics import base_metric
def update_state_on_cpu(y_true, y_pred, sample_weight=None):
    with tf.device('/cpu:0'):
        return obj_update_state(y_true, y_pred, sample_weight)