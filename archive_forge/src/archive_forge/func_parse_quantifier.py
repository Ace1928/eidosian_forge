import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_quantifier(source, info, ch):
    """Parses a quantifier."""
    q = _QUANTIFIERS.get(ch)
    if q:
        return q
    if ch == '{':
        counts = parse_limited_quantifier(source)
        if counts:
            return counts
    return None