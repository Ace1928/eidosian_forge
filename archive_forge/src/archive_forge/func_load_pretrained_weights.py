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
def load_pretrained_weights(self):
    path = keras.utils.get_file('bert_checkpoint', constants.BERT_CHECKPOINT_PATH, extract=True)
    path = os.path.join(os.path.abspath(os.path.join(path, os.pardir)), 'bert', 'bert_ckpt-1')
    checkpoint = tf.train.Checkpoint(model=self)
    checkpoint.restore(path).assert_consumed()