import networkx as nx
def edges_from(node):
    for e in G.edges(node, **kwds):
        yield (e + (FORWARD,))
    for e in G.in_edges(node, **kwds):
        yield (e + (REVERSE,))