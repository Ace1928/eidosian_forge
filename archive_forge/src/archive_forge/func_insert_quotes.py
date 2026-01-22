import re
import warnings
from enum import Enum
from math import gcd
def insert_quotes(s, d):
    """Inverse of eliminate_quotes.
    """
    for k, v in d.items():
        kind = k[:k.find('@')]
        if kind:
            kind += '_'
        s = s.replace(k, kind + v)
    return s