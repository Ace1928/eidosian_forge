import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def inRange(self, location):
    """Return TRUE if the given location is within the buffer"""
    return self._currentIndex + location < len(self._buffer)