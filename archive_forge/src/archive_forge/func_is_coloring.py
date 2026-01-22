from collections import defaultdict
import networkx as nx
@nx._dispatch
def is_coloring(G, coloring):
    """Determine if the coloring is a valid coloring for the graph G."""
    return all((coloring[s] != coloring[d] for s, d in G.edges))