import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
def in_(self, other):
    """Does self appear in other?"""
    return self in FoldedCase(other)