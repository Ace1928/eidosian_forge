from datetime import date, datetime, timedelta
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_attNameStrange_timdelta_zero_timeRespecting_returnsTrue(self):
    G1 = self.provide_g1_topology()
    temporal_name = 'strange_name'
    G1 = put_same_time(G1, temporal_name)
    G2 = self.provide_g2_path_3edges()
    d = timedelta()
    gm = iso.TimeRespectingGraphMatcher(G1, G2, temporal_name, d)
    assert gm.subgraph_is_isomorphic()