import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def is_octal(string):
    """Checks whether a string is octal."""
    return all((ch in OCT_DIGITS for ch in string))