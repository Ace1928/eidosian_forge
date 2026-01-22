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
def encoding_aliases(self) -> List[str]:
    """
        Encoding name are known by many name, using this could help when searching for IBM855 when it's listed as CP855.
        """
    also_known_as = []
    for u, p in aliases.items():
        if self.encoding == u:
            also_known_as.append(p)
        elif self.encoding == p:
            also_known_as.append(u)
    return also_known_as