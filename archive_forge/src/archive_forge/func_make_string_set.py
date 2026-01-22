import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def make_string_set(info, name):
    """Makes a string set."""
    return StringSet(info, name, case_flags=make_case_flags(info))