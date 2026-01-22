import tree
from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.utils import backend_utils
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
@classmethod
def integer_hashed(cls, num_bins, output_mode='one_hot', name=None):
    name = name or auto_name('integer_hashed')
    preprocessor = layers.Hashing(name=f'{name}_preprocessor', num_bins=num_bins)
    return Feature(dtype='int32', preprocessor=preprocessor, output_mode=output_mode)