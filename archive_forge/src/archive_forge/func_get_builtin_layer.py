import threading
import tensorflow.compat.v2 as tf
from keras.src.engine import base_layer
from keras.src.engine import input_layer
from keras.src.engine import input_spec
from keras.src.layers import activation
from keras.src.layers import attention
from keras.src.layers import convolutional
from keras.src.layers import core
from keras.src.layers import locally_connected
from keras.src.layers import merging
from keras.src.layers import pooling
from keras.src.layers import regularization
from keras.src.layers import reshaping
from keras.src.layers import rnn
from keras.src.layers.normalization import batch_normalization
from keras.src.layers.normalization import batch_normalization_v1
from keras.src.layers.normalization import group_normalization
from keras.src.layers.normalization import layer_normalization
from keras.src.layers.normalization import unit_normalization
from keras.src.layers.preprocessing import category_encoding
from keras.src.layers.preprocessing import discretization
from keras.src.layers.preprocessing import hashed_crossing
from keras.src.layers.preprocessing import hashing
from keras.src.layers.preprocessing import image_preprocessing
from keras.src.layers.preprocessing import integer_lookup
from keras.src.layers.preprocessing import (
from keras.src.layers.preprocessing import string_lookup
from keras.src.layers.preprocessing import text_vectorization
from keras.src.layers.rnn import cell_wrappers
from keras.src.layers.rnn import gru
from keras.src.layers.rnn import lstm
from keras.src.metrics import base_metric
from keras.src.saving import serialization_lib
from keras.src.saving.legacy import serialization as legacy_serialization
from keras.src.saving.legacy.saved_model import json_utils
from keras.src.utils import generic_utils
from keras.src.utils import tf_inspect as inspect
from tensorflow.python.util.tf_export import keras_export
def get_builtin_layer(class_name):
    """Returns class if `class_name` is registered, else returns None."""
    if not hasattr(LOCAL, 'ALL_OBJECTS'):
        populate_deserializable_objects()
    return LOCAL.ALL_OBJECTS.get(class_name)