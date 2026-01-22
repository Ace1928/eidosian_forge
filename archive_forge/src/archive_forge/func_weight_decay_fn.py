import tensorflow as tf
from keras.src import backend
from keras.src.backend.common import KerasVariable
from keras.src.optimizers import base_optimizer
def weight_decay_fn(variable):
    if self._use_weight_decay(variable):
        lr = tf.cast(self.learning_rate, variable.dtype)
        wd = tf.cast(self.weight_decay, variable.dtype)
        variable.assign_sub(variable * wd * lr)