import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_property_name(source):
    """Parses a property name, which may be qualified."""
    name = source.get_while(PROPERTY_NAME_PART)
    saved_pos = source.pos
    ch = source.get()
    if ch and ch in ':=':
        prop_name = name
        name = source.get_while(ALNUM | set(' &_-./')).strip()
        if name:
            saved_pos = source.pos
        else:
            prop_name, name = (None, prop_name)
    else:
        prop_name = None
    source.pos = saved_pos
    return (prop_name, name)