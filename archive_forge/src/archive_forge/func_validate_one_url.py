from __future__ import annotations
import contextlib
import functools
import urllib.request
from typing import Union, List, Optional
from typing_extensions import Annotated
from pydantic import (
@functools.lru_cache(maxsize=128)
def validate_one_url(url: str) -> bool:
    """
    Validate that the url is valid
    """
    with contextlib.suppress(Exception):
        urllib.request.urlopen(url, timeout=0.5)
        return True
    return False