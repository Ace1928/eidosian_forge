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
def get_vocabulary(self, include_special_tokens=True):
    """Returns the current vocabulary of the layer.

        Args:
          include_special_tokens: If True, the returned vocabulary will include
            mask and OOV tokens, and a term's index in the vocabulary will equal
            the term's index when calling the layer. If False, the returned
            vocabulary will not include any mask or OOV tokens.
        """
    if self.lookup_table.size() == 0:
        vocab, indices = ([], [])
    else:
        keys, values = self.lookup_table.export()
        vocab, indices = (values, keys) if self.invert else (keys, values)
        vocab, indices = (self._tensor_vocab_to_numpy(vocab), indices.numpy())
    lookup = collections.defaultdict(lambda: self.oov_token, zip(indices, vocab))
    vocab = [lookup[x] for x in range(self.vocabulary_size())]
    if self.mask_token is not None and self.output_mode == INT:
        vocab[0] = self.mask_token
    if not include_special_tokens:
        vocab = vocab[self._token_start_index():]
    return vocab