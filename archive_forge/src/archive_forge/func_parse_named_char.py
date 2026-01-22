import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_named_char(source, info, in_set):
    """Parses a named character."""
    saved_pos = source.pos
    if source.match('{'):
        name = source.get_while(NAMED_CHAR_PART)
        if source.match('}'):
            try:
                value = unicodedata.lookup(name)
                return make_character(info, ord(value), in_set)
            except KeyError:
                raise error('undefined character name', source.string, source.pos)
    source.pos = saved_pos
    return make_character(info, ord('N'), in_set)