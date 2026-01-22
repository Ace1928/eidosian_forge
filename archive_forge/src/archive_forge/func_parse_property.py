import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_property(source, info, positive, in_set):
    """Parses a Unicode property."""
    saved_pos = source.pos
    ch = source.get()
    if ch == '{':
        negate = source.match('^')
        prop_name, name = parse_property_name(source)
        if source.match('}'):
            prop = lookup_property(prop_name, name, positive != negate, source)
            return make_property(info, prop, in_set)
    elif ch and ch in 'CLMNPSZ':
        prop = lookup_property(None, ch, positive, source)
        return make_property(info, prop, in_set)
    source.pos = saved_pos
    ch = 'p' if positive else 'P'
    return make_character(info, ord(ch), in_set)