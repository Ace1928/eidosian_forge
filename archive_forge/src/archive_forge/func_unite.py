import functools
import re
import os
def unite(iterable):
    """Turns a two dimensional array into a one dimensional."""
    return set((typ for types in iterable for typ in types))