import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_set_member(source, info):
    """Parses a member in a character set."""
    start = parse_set_item(source, info)
    saved_pos1 = source.pos
    if not isinstance(start, Character) or not start.positive or (not source.match('-')):
        return start
    version = info.flags & _ALL_VERSIONS or DEFAULT_VERSION
    saved_pos2 = source.pos
    if version == VERSION1 and source.match('-'):
        source.pos = saved_pos1
        return start
    if source.match(']'):
        source.pos = saved_pos2
        return SetUnion(info, [start, Character(ord('-'))])
    end = parse_set_item(source, info)
    if not isinstance(end, Character) or not end.positive:
        return SetUnion(info, [start, Character(ord('-')), end])
    if start.value > end.value:
        raise error('bad character range', source.string, source.pos)
    if start.value == end.value:
        return start
    return Range(start.value, end.value)