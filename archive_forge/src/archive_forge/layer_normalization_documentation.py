import tensorflow.compat.v2 as tf
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import utils
from keras.src.engine.base_layer import Layer
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Returns false if fused implementation cannot be used.

        Check if the axis is contiguous and can be collapsed into the last axis.
        The self.axis is assumed to have no duplicates.
        