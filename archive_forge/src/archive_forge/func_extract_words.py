import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def extract_words(self, chars: list) -> list:
    words = list((word for word, word_chars in self.iter_extract_tuples(chars)))
    return words