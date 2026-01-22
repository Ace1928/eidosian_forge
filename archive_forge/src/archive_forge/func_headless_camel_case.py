import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
def headless_camel_case(self):
    words = iter(self)
    first = next(words).lower()
    new_words = itertools.chain((first,), WordSet(words).camel_case())
    return ''.join(new_words)