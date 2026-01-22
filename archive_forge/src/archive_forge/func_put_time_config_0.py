from datetime import date, datetime, timedelta
import networkx as nx
from networkx.algorithms import isomorphism as iso
def put_time_config_0(G, att_name):
    G[0][1][att_name] = date(2015, 1, 2)
    G[0][2][att_name] = date(2015, 1, 2)
    G[1][2][att_name] = date(2015, 1, 3)
    G[1][3][att_name] = date(2015, 1, 1)
    G[2][4][att_name] = date(2015, 1, 1)
    G[3][4][att_name] = date(2015, 1, 3)
    G[4][5][att_name] = date(2015, 1, 3)
    return G