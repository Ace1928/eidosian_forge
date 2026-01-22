import networkx as nx
from collections import deque
def incomparable(self, x):
    """
        Return the elements which are not comparable to x.

        >>> G = Digraph([(0,1),(1,2),(2,4),(0,3),(3,4)])
        >>> P = Poset(G)
        >>> sorted(P.incomparable(3))
        [1, 2]
        """
    return self.elements - self.smaller[x] - self.larger[x] - set([x])