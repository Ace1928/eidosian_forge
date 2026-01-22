import pytest
import networkx as nx
def is_lobster(g):
    """
            A tree is a lobster if it has the property that the removal of leaf
            nodes leaves a caterpillar graph (Gallian 2007)
            ref: http://mathworld.wolfram.com/LobsterGraph.html
            """
    non_leafs = [n for n in g if g.degree(n) > 1]
    return is_caterpillar(g.subgraph(non_leafs))