from __future__ import unicode_literals
import collections
import copy
import functools
import json
import sys
from jsonpointer import JsonPointer, JsonPointerException
def multidict(ordered_pairs):
    """Convert duplicate keys values to lists."""
    mdict = collections.defaultdict(list)
    for key, value in ordered_pairs:
        mdict[key].append(value)
    return dict(((key, values[0] if len(values) == 1 else values) for key, values in mdict.items()))