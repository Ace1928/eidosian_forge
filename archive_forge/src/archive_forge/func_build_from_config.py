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
def build_from_config(self, config):
    for name in config.keys():
        preprocessor = self.features[name].preprocessor
        if not preprocessor.built:
            preprocessor.build_from_config(config[name])
    self._is_adapted = True