import collections
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer_utils
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.saving.legacy.saved_model import layer_serialization
from keras.src.utils import layer_utils
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
def save_assets(self, dir_path):
    if self.input_vocabulary:
        return
    vocabulary = self.get_vocabulary(include_special_tokens=True)
    vocabulary_filepath = tf.io.gfile.join(dir_path, 'vocabulary.txt')
    with open(vocabulary_filepath, 'w') as f:
        f.write('\n'.join([str(w) for w in vocabulary]))