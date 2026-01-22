import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_call_group(source, info, ch, pos):
    """Parses a call to a group."""
    if ch == 'R':
        group = '0'
    else:
        group = ch + source.get_while(DIGITS)
    source.expect(')')
    return CallGroup(info, group, pos)