import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_set_imp_union(source, info):
    """Parses a set implicit union ([xy])."""
    version = info.flags & _ALL_VERSIONS or DEFAULT_VERSION
    items = [parse_set_member(source, info)]
    while True:
        saved_pos = source.pos
        if source.match(']'):
            source.pos = saved_pos
            break
        if version == VERSION1 and any((source.match(op) for op in SET_OPS)):
            source.pos = saved_pos
            break
        items.append(parse_set_member(source, info))
    if len(items) == 1:
        return items[0]
    return SetUnion(info, items)