import random
import sys
from . import Nodes
def has_support(self, node=None):
    """Return True if any of the nodes has data.support != None."""
    for n in self._walk(node):
        if self.node(n).data.support:
            return True
    else:
        return False