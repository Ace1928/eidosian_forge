import copy
import json
import os
import re
import warnings
from collections import UserDict
from collections.abc import Mapping, Sized
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
def word_to_tokens(self, batch_or_word_index: int, word_index: Optional[int]=None, sequence_index: int=0) -> Optional[TokenSpan]:
    """
        Get the encoded token span corresponding to a word in a sequence of the batch.

        Token spans are returned as a [`~tokenization_utils_base.TokenSpan`] with:

        - **start** -- Index of the first token.
        - **end** -- Index of the token following the last token.

        Can be called as:

        - `self.word_to_tokens(word_index, sequence_index: int = 0)` if batch size is 1
        - `self.word_to_tokens(batch_index, word_index, sequence_index: int = 0)` if batch size is greater or equal to
          1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_word_index (`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the word in the sequence.
            word_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided word index belongs to.

        Returns:
            ([`~tokenization_utils_base.TokenSpan`], *optional*): Span of tokens in the encoded sequence. Returns
            `None` if no tokens correspond to the word. This can happen especially when the token is a special token
            that has been used to format the tokenization. For example when we add a class token at the very beginning
            of the tokenization.
        """
    if not self._encodings:
        raise ValueError('word_to_tokens() is not available when using Python based tokenizers')
    if word_index is not None:
        batch_index = batch_or_word_index
    else:
        batch_index = 0
        word_index = batch_or_word_index
    if batch_index < 0:
        batch_index = self._batch_size + batch_index
    if word_index < 0:
        word_index = self._seq_len + word_index
    span = self._encodings[batch_index].word_to_tokens(word_index, sequence_index)
    return TokenSpan(*span) if span is not None else None