import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_call_named_group(source, info, pos):
    """Parses a call to a named group."""
    group = parse_name(source)
    source.expect(')')
    return CallGroup(info, group, pos)