import networkx as nx
from networkx.utils import py_random_state
def choose_peer_pref_attach(self, node_list):
    """Pick a node with a probability weighted by its peer degree.

        Pick a node from node_list with preferential attachment
        computed only on their peer degree
        """
    d = {}
    for n in node_list:
        d[n] = self.G.nodes[n]['peers']
    return choose_pref_attach(d, self.seed)