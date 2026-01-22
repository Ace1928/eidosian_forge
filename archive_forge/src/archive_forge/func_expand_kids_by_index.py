from copy import deepcopy
from collections import OrderedDict
def expand_kids_by_index(self, *indices):
    """Expand (inline) children at the given indices"""
    for i in sorted(indices, reverse=True):
        kid = self.children[i]
        self.children[i:i + 1] = kid.children