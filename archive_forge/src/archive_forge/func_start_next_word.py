import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def start_next_word(new_char=None):
    nonlocal current_word
    if current_word:
        yield current_word
    current_word = [] if new_char is None else [new_char]