import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_cost_equation(source, constraints):
    """Parses a cost equation."""
    if 'cost' in constraints:
        raise error('more than one cost equation', source.string, source.pos)
    cost = {}
    parse_cost_term(source, cost)
    while source.match('+'):
        parse_cost_term(source, cost)
    max_inc = parse_fuzzy_compare(source)
    if max_inc is None:
        raise ParseError()
    max_cost = int(parse_count(source))
    if not max_inc:
        max_cost -= 1
    if max_cost < 0:
        raise error('bad fuzzy cost limit', source.string, source.pos)
    cost['max'] = max_cost
    constraints['cost'] = cost