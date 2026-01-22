import networkx as nx
from networkx.utils import py_random_state
def choose_node_pref_attach(self, node_list):
    """Pick a node with a probability weighted by its degree.

        Pick a node from node_list with preferential attachment
        computed on their degree
        """
    degs = dict(self.G.degree(node_list))
    return choose_pref_attach(degs, self.seed)