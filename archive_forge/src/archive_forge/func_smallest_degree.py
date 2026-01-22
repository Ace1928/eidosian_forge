import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering
def smallest_degree(G):
    deg, node = min(((d, n) for n, d in G.degree()))
    return node