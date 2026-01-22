import importlib
import logging
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import List, Optional, Set, Tuple, Union
from _multibytecodec import MultibyteIncrementalDecoder  # type: ignore
from .constant import (
@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_accentuated(character: str) -> bool:
    try:
        description = unicodedata.name(character)
    except ValueError:
        return False
    return 'WITH GRAVE' in description or 'WITH ACUTE' in description or 'WITH CEDILLA' in description or ('WITH DIAERESIS' in description) or ('WITH CIRCUMFLEX' in description) or ('WITH TILDE' in description)