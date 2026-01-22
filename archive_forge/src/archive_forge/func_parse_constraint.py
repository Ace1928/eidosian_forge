import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_constraint(source, constraints, ch):
    """Parses a constraint."""
    if ch not in 'deis':
        raise ParseError()
    if ch in constraints:
        raise ParseError()
    return ch