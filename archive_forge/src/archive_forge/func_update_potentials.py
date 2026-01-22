from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
def update_potentials(self, i, p, q):
    """
        Update the potentials of the nodes in the subtree rooted at a node
        q connected to its parent p by an edge i.
        """
    if q == self.edge_targets[i]:
        d = self.node_potentials[p] - self.edge_weights[i] - self.node_potentials[q]
    else:
        d = self.node_potentials[p] + self.edge_weights[i] - self.node_potentials[q]
    for q in self.trace_subtree(q):
        self.node_potentials[q] += d