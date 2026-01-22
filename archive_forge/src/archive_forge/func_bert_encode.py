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
def bert_encode(self, input_tensor):
    sentence = self.get_encoded_sentence(input_tensor)
    cls = [self.tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence.shape[0]
    input_word_ids = tf.concat([cls, sentence], axis=-1).to_tensor()
    if input_word_ids.shape[-1] > self.max_sequence_length:
        input_word_ids = input_word_ids[..., :self.max_sequence_length]
    return input_word_ids