import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_lookaround(source, info, behind, positive):
    """Parses a lookaround."""
    saved_flags = info.flags
    try:
        subpattern = _parse_pattern(source, info)
        source.expect(')')
    finally:
        info.flags = saved_flags
        source.ignore_space = bool(info.flags & VERBOSE)
    return LookAround(behind, positive, subpattern)