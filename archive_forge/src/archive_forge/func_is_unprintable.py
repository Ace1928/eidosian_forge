import importlib
import logging
import unicodedata
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import Generator, List, Optional, Set, Tuple, Union
from _multibytecodec import MultibyteIncrementalDecoder
from .constant import (
@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_unprintable(character: str) -> bool:
    return character.isspace() is False and character.isprintable() is False and (character != '\x1a') and (character != '\ufeff')