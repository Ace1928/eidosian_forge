from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for
def hide_edge(n, nbr, d):
    if n not in enodes or nbr not in enodes:
        return wt(n, nbr, d)
    return None