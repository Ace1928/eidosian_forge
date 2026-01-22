import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def lower_key(key):
    if isinstance(key, (bytes, str)):
        return key.lower()
    if isinstance(key, Iterable):
        return type(key)(map(lower_key, key))
    return key