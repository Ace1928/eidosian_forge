import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def contains_group(self):
    return any((s.contains_group() for s in self.items))