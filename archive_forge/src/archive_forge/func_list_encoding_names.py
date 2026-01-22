from __future__ import annotations
import functools
import importlib
import pkgutil
import threading
from typing import Any, Callable, Optional, Sequence
import tiktoken_ext
from tiktoken.core import Encoding
def list_encoding_names() -> list[str]:
    with _lock:
        if ENCODING_CONSTRUCTORS is None:
            _find_constructors()
            assert ENCODING_CONSTRUCTORS is not None
        return list(ENCODING_CONSTRUCTORS)