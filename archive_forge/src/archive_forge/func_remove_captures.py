import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def remove_captures(self):
    self.items = [s.remove_captures() for s in self.items]
    return self