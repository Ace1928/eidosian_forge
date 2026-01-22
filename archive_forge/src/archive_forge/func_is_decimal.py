import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def is_decimal(string):
    """Checks whether a string is decimal."""
    return all((ch in DIGITS for ch in string))