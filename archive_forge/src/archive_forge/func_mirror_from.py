from __future__ import annotations
import contextlib
import logging
import math
import os
import pathlib
import re
import sys
import tempfile
from functools import partial
from hashlib import md5
from importlib.metadata import version
from typing import (
from urllib.parse import urlsplit
def mirror_from(origin_name: str, methods: Iterable[str]) -> Callable[[type[T]], type[T]]:
    """Mirror attributes and methods from the given
    origin_name attribute of the instance to the
    decorated class"""

    def origin_getter(method: str, self: Any) -> Any:
        origin = getattr(self, origin_name)
        return getattr(origin, method)

    def wrapper(cls: type[T]) -> type[T]:
        for method in methods:
            wrapped_method = partial(origin_getter, method)
            setattr(cls, method, property(wrapped_method))
        return cls
    return wrapper