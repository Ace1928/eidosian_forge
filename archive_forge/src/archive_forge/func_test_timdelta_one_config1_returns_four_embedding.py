from datetime import date, datetime, timedelta
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_timdelta_one_config1_returns_four_embedding(self):
    G1 = self.provide_g1_topology()
    temporal_name = 'date'
    G1 = put_time_config_1(G1, temporal_name)
    G2 = self.provide_g2_path_3edges()
    d = timedelta(days=1)
    gm = iso.TimeRespectingGraphMatcher(G1, G2, temporal_name, d)
    count_match = len(list(gm.subgraph_isomorphisms_iter()))
    assert count_match == 4