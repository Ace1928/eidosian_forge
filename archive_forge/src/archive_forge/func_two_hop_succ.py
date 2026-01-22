import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
def two_hop_succ(self, Gx, Gx_node, core_x, succ):
    """
        The successors of the ego node.
        """
    return all((self.one_hop(Gx, s, core_x, self.preds(Gx, core_x, s, Gx_node), self.succs(Gx, core_x, s)) for s in succ))