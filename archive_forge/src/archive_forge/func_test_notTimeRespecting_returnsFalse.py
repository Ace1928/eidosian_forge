from datetime import date, datetime, timedelta
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_notTimeRespecting_returnsFalse(self):
    G1 = self.provide_g1_topology()
    temporal_name = 'date'
    G1 = put_sequence_time(G1, temporal_name)
    G2 = self.provide_g2_path_3edges()
    d = timedelta()
    gm = iso.TimeRespectingGraphMatcher(G1, G2, temporal_name, d)
    assert not gm.subgraph_is_isomorphic()