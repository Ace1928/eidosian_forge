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
def set_vocabulary(self, vocabulary, idf_weights=None):
    """Sets vocabulary (and optionally document frequency) for this layer.

        This method sets the vocabulary and idf weights for this layer directly,
        instead of analyzing a dataset through `adapt`. It should be used
        whenever the vocab (and optionally document frequency) information is
        already known.  If vocabulary data is already present in the layer, this
        method will replace it.

        Args:
          vocabulary: Either an array or a string path to a text file. If
            passing an array, can pass a tuple, list, 1D numpy array, or 1D
            tensor containing the vocbulary terms. If passing a file path, the
            file should contain one line per term in the vocabulary.
          idf_weights: A tuple, list, 1D numpy array, or 1D tensor of inverse
            document frequency weights with equal length to vocabulary. Must be
            set if `output_mode` is `"tf_idf"`. Should not be set otherwise.

        Raises:
          ValueError: If there are too many inputs, the inputs do not match, or
            input data is missing.
          RuntimeError: If the vocabulary cannot be set when this function is
            called. This happens when `"multi_hot"`, `"count"`, and `"tf_idf"`
            modes, if `pad_to_max_tokens` is False and the layer itself has
            already been called.
          RuntimeError: If a tensor vocabulary is passed outside of eager
            execution.
        """
    if self.output_mode == TF_IDF:
        if idf_weights is None:
            raise ValueError('`idf_weights` must be set if output_mode is TF_IDF')
    elif idf_weights is not None:
        raise ValueError(f"`idf_weights` should only be set if output_mode is `'tf_idf'`. Received: output_mode={self.output_mode} and idf_weights={idf_weights}")
    if isinstance(vocabulary, str):
        if not tf.io.gfile.exists(vocabulary):
            raise ValueError(f'Vocabulary file {vocabulary} does not exist.')
        if self.output_mode == TF_IDF:
            raise ValueError("output_mode `'tf_idf'` does not support loading a vocabulary from file.")
        self.lookup_table = self._lookup_table_from_file(vocabulary)
        self._record_vocabulary_size()
        return
    if not tf.executing_eagerly() and (tf.is_tensor(vocabulary) or tf.is_tensor(idf_weights)):
        raise RuntimeError('Cannot set a tensor vocabulary on {} layer {} when not executing eagerly. Create this layer or call `set_vocabulary` outside of any `tf.function`s and with eager execution enabled.'.format(self.__class__.__name__, self.name))
    if tf.is_tensor(vocabulary):
        vocabulary = self._tensor_vocab_to_numpy(vocabulary)
    elif isinstance(vocabulary, (list, tuple)):
        vocabulary = np.array(vocabulary)
    if tf.is_tensor(idf_weights):
        idf_weights = idf_weights.numpy()
    elif isinstance(idf_weights, (list, tuple)):
        idf_weights = np.array(idf_weights)
    if vocabulary.size == 0:
        raise ValueError(f'Cannot set an empty vocabulary, you passed {vocabulary}.')
    oov_start = self._oov_start_index()
    token_start = self._token_start_index()
    special_tokens = [self.mask_token] * oov_start + [self.oov_token] * self.num_oov_indices
    found_special_tokens = np.array_equal(special_tokens, vocabulary[:token_start])
    if found_special_tokens:
        tokens = vocabulary[token_start:]
    else:
        tokens = vocabulary
    repeated_tokens = self._find_repeated_tokens(tokens)
    if repeated_tokens:
        raise ValueError('The passed vocabulary has at least one repeated term. Please uniquify your dataset. The repeated terms are {}'.format(repeated_tokens))
    if self.mask_token is not None and self.mask_token in tokens:
        mask_index = np.argwhere(vocabulary == self.mask_token)[-1]
        raise ValueError(f'Found reserved mask token at unexpected location in `vocabulary`. Note that passed `vocabulary` does not need to include the OOV and mask tokens. Either remove all mask and OOV tokens, or include them only at the start of the vocabulary in precisely this order: {special_tokens}. Received: mask_token={self.mask_token} at vocabulary index {mask_index}')
    if self.oov_token is not None and self.invert and (self.oov_token in tokens):
        oov_index = np.argwhere(vocabulary == self.oov_token)[-1]
        raise ValueError(f'Found reserved OOV token at unexpected location in `vocabulary`. Note that passed `vocabulary` does not need to include the OOV and mask tokens. Either remove all mask and OOV tokens, or include them only at the start of the vocabulary in precisely this order: {special_tokens}. Received: oov_token={self.oov_token} at vocabulary index {oov_index}')
    new_vocab_size = token_start + len(tokens)
    if self.max_tokens is not None and new_vocab_size > self.max_tokens:
        raise ValueError('Attempted to set a vocabulary larger than the maximum vocab size. Passed vocab size is {}, max vocab size is {}.'.format(new_vocab_size, self.max_tokens))
    self.lookup_table = self._lookup_table_from_tokens(tokens)
    self._record_vocabulary_size()
    if self.output_mode == TF_IDF and idf_weights is not False:
        if len(vocabulary) != len(idf_weights):
            raise ValueError('`idf_weights` must be the same length as vocabulary. len(idf_weights) is {}, len(vocabulary) is {}'.format(len(vocabulary), len(idf_weights)))
        idf_weights = self._convert_to_ndarray(idf_weights)
        if idf_weights.ndim != 1:
            raise ValueError('TF-IDF data must be a 1-index array, but received {}'.format(type(idf_weights)))
        if found_special_tokens:
            front_padding = 0
            front_padding_value = 0
        else:
            front_padding = token_start
            front_padding_value = np.average(idf_weights)
        back_padding_value = 0
        if self.pad_to_max_tokens and self.max_tokens is not None:
            back_padding = self.max_tokens - front_padding - len(idf_weights)
        else:
            back_padding = 0
        weights = np.pad(idf_weights, (front_padding, back_padding), 'constant', constant_values=(front_padding_value, back_padding_value))
        weights = tf.convert_to_tensor(weights, dtype=self.compute_dtype)
        self.idf_weights.assign(weights)
        self.idf_weights_const = self.idf_weights.value()