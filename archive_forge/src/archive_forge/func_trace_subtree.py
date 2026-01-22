from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
def trace_subtree(self, p):
    """
        Yield the nodes in the subtree rooted at a node p.
        """
    yield p
    l = self.last_descendent_dft[p]
    while p != l:
        p = self.next_node_dft[p]
        yield p