import networkx as nx
from collections import deque
def largest(self):
    """
        Return the subset of maximal elements.

        >>> G = Digraph([(0,1),(1,2),(2,4),(0,3),(3,4)])
        >>> P = Poset(G)
        >>> sorted(P.largest())
        [4]
        """
    return frozenset([x for x in self if not self.larger[x]])