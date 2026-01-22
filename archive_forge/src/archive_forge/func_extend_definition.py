from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def extend_definition(self, name, dnode_or_dnodes):
    if self.deadcode:
        return
    DefUseChains.add_to_definition(self._definitions[-1], name, dnode_or_dnodes)