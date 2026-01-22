from collections import namedtuple
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Generator, List, Sequence, Set, Tuple
import numba
import numpy as np
from interegular.fsm import FSM, Alphabet, OblivionError, anything_else
from numba.typed.typedobjectutils import _nonoptional
@lru_cache
def reduced_vocabulary(tokenizer: 'Tokenizer'):
    """Create a map from decoded vocabulary tokens to lists of equivalent token ids."""
    vocabulary = numba.typed.Dict.empty(numba.types.string, numba.types.ListType(numba.int64))
    empty_token_ids = set()
    for token, token_idx in tokenizer.vocabulary.items():
        if token in tokenizer.special_tokens:
            continue
        token_str = tokenizer.convert_token_to_string(token)
        if token_str:
            vocabulary.setdefault(token_str, numba.typed.List.empty_list(numba.int64)).append(numba.int64(token_idx))
        else:
            empty_token_ids.add(numba.int64(token_idx))
    return (vocabulary, empty_token_ids)