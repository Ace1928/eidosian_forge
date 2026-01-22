from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Collection, Literal, NoReturn, Optional, Union
import regex
from tiktoken import _tiktoken
def token_byte_values(self) -> list[bytes]:
    """Returns the list of all token byte values."""
    return self._core_bpe.token_byte_values()