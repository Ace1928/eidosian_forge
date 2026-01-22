import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def make_case_flags(info):
    """Makes the case flags."""
    flags = info.flags & CASE_FLAGS
    if info.flags & ASCII:
        flags &= ~FULLCASE
    return flags