from datetime import datetime, timedelta
import pytest
import networkx as nx
def test_common_graph_with_given_attributes(self):
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    G.add_edge(4, 2)
    G.add_edge(4, 0)
    G.add_edge(4, 1)
    G.add_edge(4, 3)
    G.add_edge(5, 2)
    G.add_edge(6, 2)
    G.add_edge(6, 4)
    G.add_edge(7, 4)
    G.add_edge(8, 4)
    G.add_edge(9, 4)
    G.add_edge(9, 1)
    G.add_edge(9, 3)
    G.add_edge(10, 4)
    node_attrs = {0: {'date': datetime(1992, 1, 1)}, 1: {'date': datetime(1992, 1, 1)}, 2: {'date': datetime(1993, 1, 1)}, 3: {'date': datetime(1993, 1, 1)}, 4: {'date': datetime(1995, 1, 1)}, 5: {'date': datetime(1997, 1, 1)}, 6: {'date': datetime(1998, 1, 1)}, 7: {'date': datetime(1999, 1, 1)}, 8: {'date': datetime(1999, 1, 1)}, 9: {'date': datetime(1998, 1, 1)}, 10: {'date': datetime(1997, 4, 1)}}
    nx.set_node_attributes(G, node_attrs)
    assert nx.cd_index(G, 4, time_delta=_delta, time='date') == 0.17