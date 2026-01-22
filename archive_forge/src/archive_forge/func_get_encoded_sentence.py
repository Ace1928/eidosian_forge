import collections
import math
import os
import re
import unicodedata
from typing import List
import numpy as np
import six
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from autokeras import constants
from autokeras.utils import data_utils
def get_encoded_sentence(self, input_tensor):
    input_array = np.array(input_tensor, dtype=object)
    sentence = tf.ragged.constant([self.encode_sentence(s[0]) for s in input_array])
    return sentence