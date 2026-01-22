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
def chaos_secondary_pass(self) -> float:
    """
        Check once again chaos in decoded text, except this time, with full content.
        Use with caution, this can be very slow.
        Notice: Will be removed in 3.0
        """
    warnings.warn('chaos_secondary_pass is deprecated and will be removed in 3.0', DeprecationWarning)
    return mess_ratio(str(self), 1.0)