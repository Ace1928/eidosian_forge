import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def make_sequence(items):
    if len(items) == 1:
        return items[0]
    return Sequence(items)