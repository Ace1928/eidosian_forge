import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_cost_term(source, cost):
    """Parses a cost equation term."""
    coeff = parse_count(source)
    ch = source.get()
    if ch not in 'dis':
        raise ParseError()
    if ch in cost:
        raise error('repeated fuzzy cost', source.string, source.pos)
    cost[ch] = int(coeff or 1)