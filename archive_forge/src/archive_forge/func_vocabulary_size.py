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
def vocabulary_size(self):
    """Gets the current size of the layer's vocabulary.

        Returns:
          The integer size of the vocabulary, including optional mask and oov
          indices.
        """
    if tf.executing_eagerly():
        return int(self.lookup_table.size().numpy()) + self._token_start_index()
    else:
        return self.lookup_table.size() + self._token_start_index()