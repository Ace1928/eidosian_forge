import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def with_flags(self, positive=None, case_flags=None, zerowidth=None):
    if positive is None:
        positive = self.positive
    else:
        positive = bool(positive)
    if case_flags is None:
        case_flags = self.case_flags
    else:
        case_flags = CASE_FLAGS_COMBINATIONS[case_flags & CASE_FLAGS]
    if zerowidth is None:
        zerowidth = self.zerowidth
    else:
        zerowidth = bool(zerowidth)
    if positive == self.positive and case_flags == self.case_flags and (zerowidth == self.zerowidth):
        return self
    return self.rebuild(positive, case_flags, zerowidth)