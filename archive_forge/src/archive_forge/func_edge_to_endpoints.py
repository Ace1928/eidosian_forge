import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def edge_to_endpoints(e):
    if swap_hor_edges and e.kind == 'horizontal':
        return (e.head, e.tail)
    return (e.tail, e.head)