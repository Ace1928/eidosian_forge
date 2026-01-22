import collections
import io
import os
import networkx as nx
from networkx.drawing import nx_pydot
def no_successors_iter(self):
    """Returns an iterator for all nodes with no successors."""
    for n in self.nodes:
        if not len(list(self.successors(n))):
            yield n