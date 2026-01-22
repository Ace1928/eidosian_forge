import warnings
from collections import Counter
from encodings.aliases import aliases
from hashlib import sha256
from json import dumps
from re import sub
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from .constant import NOT_PRINTABLE_PATTERN, TOO_BIG_SEQUENCE
from .md import mess_ratio
from .utils import iana_name, is_multi_byte_encoding, unicode_range
@property
def w_counter(self) -> Counter:
    """
        Word counter instance on decoded text.
        Notice: Will be removed in 3.0
        """
    warnings.warn('w_counter is deprecated and will be removed in 3.0', DeprecationWarning)
    string_printable_only = sub(NOT_PRINTABLE_PATTERN, ' ', str(self).lower())
    return Counter(string_printable_only.split())