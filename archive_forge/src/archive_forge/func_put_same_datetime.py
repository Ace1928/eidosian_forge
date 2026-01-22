from datetime import date, datetime, timedelta
import networkx as nx
from networkx.algorithms import isomorphism as iso
def put_same_datetime(G, att_name):
    for e in G.edges(data=True):
        e[2][att_name] = datetime(2015, 1, 1)
    return G