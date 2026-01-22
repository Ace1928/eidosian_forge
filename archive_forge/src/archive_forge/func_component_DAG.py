import networkx as nx
from collections import deque
def component_DAG(self):
    """
        Return the acyclic digraph whose vertices are the strong
        components of this digraph.  Two components are joined by an
        edge if this digraph has an edge from one component to the
        other.

        >>> G = Digraph([(0,1),(0,2),(1,2),(2,3),(3,1)])
        >>> C = G.component_DAG()
        >>> sorted(C.vertices) == [frozenset([1, 2, 3]), frozenset([0])]
        True
        >>> [tuple(e) for e in C.edges] == [(frozenset([0]), frozenset([1, 2, 3]))]
        True
        """
    return StrongConnector(self).DAG()