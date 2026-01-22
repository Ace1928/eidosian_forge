import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def switches(self, swap_hor_edges):
    """
        Returns a list of (index, source|sink, -1|1).
        """

    def edge_to_endpoints(e):
        if swap_hor_edges and e.kind == 'horizontal':
            return (e.head, e.tail)
        return (e.tail, e.head)
    ans = []
    for i, (e0, v0) in enumerate(self):
        t0, h0 = edge_to_endpoints(e0)
        t1, h1 = edge_to_endpoints(self[i + 1][0])
        if t0 == t1 == v0:
            ans.append(LabeledFaceVertex(i, 'source', self.turns[i]))
        elif h0 == h1 == v0:
            ans.append(LabeledFaceVertex(i, 'sink', self.turns[i]))
    return ans