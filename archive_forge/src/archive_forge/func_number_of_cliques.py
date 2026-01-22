from collections import defaultdict, deque
from itertools import chain, combinations, islice
import networkx as nx
from networkx.utils import not_implemented_for
def number_of_cliques(G, nodes=None, cliques=None):
    """Returns the number of maximal cliques for each node.

    Returns a single or list depending on input nodes.
    Optional list of cliques can be input if already computed.
    """
    if cliques is None:
        cliques = list(find_cliques(G))
    if nodes is None:
        nodes = list(G.nodes())
    if not isinstance(nodes, list):
        v = nodes
        numcliq = len([1 for c in cliques if v in c])
    else:
        numcliq = {}
        for v in nodes:
            numcliq[v] = len([1 for c in cliques if v in c])
    return numcliq