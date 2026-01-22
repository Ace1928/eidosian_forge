from datetime import datetime, timedelta
import pytest
import networkx as nx
def test_time_timedelta_compatibility(self):
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    G.add_edge(4, 2)
    G.add_edge(4, 0)
    G.add_edge(4, 3)
    G.add_edge(6, 4)
    G.add_edge(7, 4)
    G.add_edge(8, 4)
    G.add_edge(9, 4)
    G.add_edge(9, 1)
    G.add_edge(10, 4)
    node_attrs = {0: {'time': 20.2}, 1: {'time': 20.2}, 2: {'time': 30.7}, 3: {'time': 30.7}, 4: {'time': 50.9}, 5: {'time': 70.1}, 6: {'time': 80.6}, 7: {'time': 90.7}, 8: {'time': 90.7}, 9: {'time': 80.6}, 10: {'time': 74.2}}
    nx.set_node_attributes(G, node_attrs)
    with pytest.raises(nx.NetworkXError, match='Addition and comparison are not supported between') as ve:
        nx.cd_index(G, 4, time_delta=_delta)