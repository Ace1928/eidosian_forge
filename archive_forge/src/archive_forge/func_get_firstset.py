import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def get_firstset(self, reverse):
    if reverse:
        pos = -1
    else:
        pos = 0
    return set([Character(self.characters[pos], case_flags=self.case_flags)])