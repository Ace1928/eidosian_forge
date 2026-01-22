import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_set_diff(source, info):
    """Parses a set difference ([x--y])."""
    items = [parse_set_imp_union(source, info)]
    while source.match('--'):
        items.append(parse_set_imp_union(source, info))
    if len(items) == 1:
        return items[0]
    return SetDiff(info, items)