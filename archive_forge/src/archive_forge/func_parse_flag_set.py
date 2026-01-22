import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_flag_set(source):
    """Parses a set of inline flags."""
    flags = 0
    try:
        while True:
            saved_pos = source.pos
            ch = source.get()
            if ch == 'V':
                ch += source.get()
            flags |= REGEX_FLAGS[ch]
    except KeyError:
        source.pos = saved_pos
    return flags