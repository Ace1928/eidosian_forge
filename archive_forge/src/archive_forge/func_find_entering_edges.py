from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
def find_entering_edges(self):
    """Yield entering edges until none can be found."""
    if self.edge_count == 0:
        return
    B = int(ceil(sqrt(self.edge_count)))
    M = (self.edge_count + B - 1) // B
    m = 0
    f = 0
    while m < M:
        l = f + B
        if l <= self.edge_count:
            edges = range(f, l)
        else:
            l -= self.edge_count
            edges = chain(range(f, self.edge_count), range(l))
        f = l
        i = min(edges, key=self.reduced_cost)
        c = self.reduced_cost(i)
        if c >= 0:
            m += 1
        else:
            if self.edge_flow[i] == 0:
                p = self.edge_sources[i]
                q = self.edge_targets[i]
            else:
                p = self.edge_targets[i]
                q = self.edge_sources[i]
            yield (i, p, q)
            m = 0