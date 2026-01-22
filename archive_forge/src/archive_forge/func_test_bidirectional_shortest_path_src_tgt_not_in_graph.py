import pytest
import networkx as nx
@pytest.mark.parametrize(('src', 'tgt'), ((8, 3), (3, 8), (8, 10), (8, 8)))
def test_bidirectional_shortest_path_src_tgt_not_in_graph(self, src, tgt):
    with pytest.raises(nx.NodeNotFound, match=f'Either source {src} or target {tgt} is not in G'):
        nx.bidirectional_shortest_path(self.cycle, src, tgt)