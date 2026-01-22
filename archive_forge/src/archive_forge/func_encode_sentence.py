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
def encode_sentence(self, s):
    """Encodes a sentence using the BERT tokenizer.

        Tokenizes, and adjusts the sentence length to the maximum sequence
        length. Some important tokens in the BERT tokenizer are:
        [UNK]: 100, [CLS]: 101, [SEP]: 102, [MASK]: 103.

        # Arguments
            s: Tensor. Raw sentence string.
        """
    tokens = list(self.tokenizer.tokenize(s))
    tokens.append('[SEP]')
    encoded_sentence = self.tokenizer.convert_tokens_to_ids(tokens)
    return encoded_sentence