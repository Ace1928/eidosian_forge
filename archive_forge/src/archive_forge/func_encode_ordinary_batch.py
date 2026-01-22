from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union
import regex
from tiktoken import _tiktoken
def encode_ordinary_batch(self, text: list[str], *, num_threads: int=8) -> list[list[int]]:
    """Encodes a list of strings into tokens, in parallel, ignoring special tokens.

        This is equivalent to `encode_batch(text, disallowed_special=())` (but slightly faster).

        ```
        >>> enc.encode_ordinary_batch(["hello world", "goodbye world"])
        [[31373, 995], [11274, 16390, 995]]
        ```
        """
    encoder = functools.partial(self.encode_ordinary)
    with ThreadPoolExecutor(num_threads) as e:
        return list(e.map(encoder, text))