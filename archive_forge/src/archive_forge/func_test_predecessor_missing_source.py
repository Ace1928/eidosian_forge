import pytest
import networkx as nx
def test_predecessor_missing_source(self):
    source = 8
    with pytest.raises(nx.NodeNotFound, match=f'Source {source} not in G'):
        nx.predecessor(self.cycle, source)