import logging
import os
import tarfile
import warnings
import zipfile
from . import _constants as C
from . import vocab
from ... import ndarray as nd
from ... import registry
from ... import base
from ...util import is_np_array
from ... import numpy as _mx_np
from ... import numpy_extension as _mx_npx
def update_token_vectors(self, tokens, new_vectors):
    """Updates embedding vectors for tokens.


        Parameters
        ----------
        tokens : str or a list of strs
            A token or a list of tokens whose embedding vector are to be updated.
        new_vectors : mxnet.ndarray.NDArray
            An NDArray to be assigned to the embedding vectors of `tokens`. Its length must be equal
            to the number of `tokens` and its width must be equal to the dimension of embeddings of
            the glossary. If `tokens` is a singleton, it must be 1-D or 2-D. If `tokens` is a list
            of multiple strings, it must be 2-D.
        """
    assert self.idx_to_vec is not None, 'The property `idx_to_vec` has not been properly set.'
    if not isinstance(tokens, list) or len(tokens) == 1:
        assert isinstance(new_vectors, nd.NDArray) and len(new_vectors.shape) in [1, 2], '`new_vectors` must be a 1-D or 2-D NDArray if `tokens` is a singleton.'
        if not isinstance(tokens, list):
            tokens = [tokens]
        if len(new_vectors.shape) == 1:
            expand_dims_fn = _mx_np.expand_dims if is_np_array() else nd.expand_dims
            new_vectors = expand_dims_fn(new_vectors, axis=0)
    else:
        assert isinstance(new_vectors, nd.NDArray) and len(new_vectors.shape) == 2, '`new_vectors` must be a 2-D NDArray if `tokens` is a list of multiple strings.'
    assert new_vectors.shape == (len(tokens), self.vec_len), 'The length of new_vectors must be equal to the number of tokens and the width ofnew_vectors must be equal to the dimension of embeddings of the glossary.'
    indices = []
    for token in tokens:
        if token in self.token_to_idx:
            indices.append(self.token_to_idx[token])
        else:
            raise ValueError('Token %s is unknown. To update the embedding vector for an unknown token, please specify it explicitly as the `unknown_token` %s in `tokens`. This is to avoid unintended updates.' % (token, self.idx_to_token[C.UNKNOWN_IDX]))
    array_fn = _mx_np.array if is_np_array() else nd.array
    self._idx_to_vec[array_fn(indices)] = new_vectors