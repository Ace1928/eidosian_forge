import importlib
from codecs import IncrementalDecoder
from collections import Counter, OrderedDict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from .assets import FREQUENCIES
from .constant import KO_NAMES, LANGUAGE_SUPPORTED_COUNT, TOO_SMALL_SEQUENCE, ZH_NAMES
from .md import is_suspiciously_successive_range
from .models import CoherenceMatches
from .utils import (
@lru_cache()
def encoding_languages(iana_name: str) -> List[str]:
    """
    Single-byte encoding language association. Some code page are heavily linked to particular language(s).
    This function does the correspondence.
    """
    unicode_ranges = encoding_unicode_range(iana_name)
    primary_range = None
    for specified_range in unicode_ranges:
        if 'Latin' not in specified_range:
            primary_range = specified_range
            break
    if primary_range is None:
        return ['Latin Based']
    return unicode_range_languages(primary_range)