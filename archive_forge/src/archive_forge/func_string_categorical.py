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
def string_categorical(cls, max_tokens=None, num_oov_indices=1, output_mode='one_hot', name=None):
    name = name or auto_name('string_categorical')
    preprocessor = layers.StringLookup(name=f'{name}_preprocessor', max_tokens=max_tokens, num_oov_indices=num_oov_indices)
    return Feature(dtype='string', preprocessor=preprocessor, output_mode=output_mode)