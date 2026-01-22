import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def optimise(self, info, reverse, in_set=False):
    items = []
    for m in self.items:
        m = m.optimise(info, reverse, in_set=True)
        if isinstance(m, SetUnion) and m.positive:
            items.extend(m.items)
        else:
            items.append(m)
    if len(items) == 1:
        i = items[0]
        return i.with_flags(positive=i.positive == self.positive, case_flags=self.case_flags, zerowidth=self.zerowidth).optimise(info, reverse, in_set)
    self.items = tuple(items)
    return self._handle_case_folding(info, in_set)