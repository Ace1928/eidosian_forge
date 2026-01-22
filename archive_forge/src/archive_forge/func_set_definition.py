from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def set_definition(self, name, dnode_or_dnodes):
    if self.deadcode:
        return
    if isinstance(dnode_or_dnodes, Def):
        self._definitions[-1][name] = ordered_set((dnode_or_dnodes,))
    else:
        self._definitions[-1][name] = ordered_set(dnode_or_dnodes)