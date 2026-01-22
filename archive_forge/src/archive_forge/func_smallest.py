import networkx as nx
from collections import deque
def smallest(self):
    """
        Return the subset of minimal elements.

        >>> G = Digraph([(0,1),(1,2),(2,4),(0,3),(3,4)])
        >>> P = Poset(G)
        >>> sorted(P.smallest())
        [0]
        """
    return frozenset([x for x in self if not self.smaller[x]])