import networkx as nx
from networkx.algorithms.approximation import min_weighted_vertex_cover
def is_cover(G, node_cover):
    return all(({u, v} & node_cover for u, v in G.edges()))