from datetime import date, datetime, timedelta
import networkx as nx
from networkx.algorithms import isomorphism as iso
def provide_g1_topology(self):
    G1 = nx.DiGraph()
    G1.add_edges_from(provide_g1_edgelist())
    return G1