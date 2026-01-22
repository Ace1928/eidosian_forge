from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union
import regex
from tiktoken import _tiktoken
def encode_single_token(self, text_or_bytes: Union[str, bytes]) -> int:
    """Encodes text corresponding to a single token to its token value.

        NOTE: this will encode all special tokens.

        Raises `KeyError` if the token is not in the vocabulary.

        ```
        >>> enc.encode_single_token("hello")
        31373
        ```
        """
    if isinstance(text_or_bytes, str):
        text_or_bytes = text_or_bytes.encode('utf-8')
    return self._core_bpe.encode_single_token(text_or_bytes)