from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def sumByElement(tokens):
    elementsList = [t[0] for t in tokens]
    duplicates = len(elementsList) > len(set(elementsList))
    if duplicates:
        ctr = defaultdict(int)
        for t in tokens:
            ctr[t[0]] += t[1]
        return ParseResults([ParseResults([k, v]) for k, v in ctr.items()])