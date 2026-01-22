import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def is_above_limit(count):
    """Checks whether a count is above the maximum."""
    return count is not None and count >= UNLIMITED