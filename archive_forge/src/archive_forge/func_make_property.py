import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def make_property(info, prop, in_set):
    """Makes a property."""
    if in_set:
        return prop
    return prop.with_flags(case_flags=make_case_flags(info))