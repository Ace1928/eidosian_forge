from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def multiplyContents(tokens):
    t = tokens[0]
    if t.subgroup:
        mult = t.mult
        for term in t.subgroup:
            term[1] *= mult
        return t.subgroup