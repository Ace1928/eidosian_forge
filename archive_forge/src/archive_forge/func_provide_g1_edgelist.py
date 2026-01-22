from datetime import date, datetime, timedelta
import networkx as nx
from networkx.algorithms import isomorphism as iso
def provide_g1_edgelist():
    return [(0, 1), (0, 2), (1, 2), (2, 4), (1, 3), (3, 4), (4, 5)]