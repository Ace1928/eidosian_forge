import pytest
import networkx as nx
def test_dual_barabasi_albert(self, m1=1, m2=4, p=0.5):
    """
        Tests that the dual BA random graph generated behaves consistently.

        Tests the exceptions are raised as expected.

        The graphs generation are repeated several times to prevent lucky shots

        """
    seeds = [42, 314, 2718]
    initial_graph = nx.complete_graph(10)
    for seed in seeds:
        BA1 = nx.barabasi_albert_graph(100, m1, seed)
        DBA1 = nx.dual_barabasi_albert_graph(100, m1, m2, 1, seed)
        assert BA1.edges() == DBA1.edges()
        BA2 = nx.barabasi_albert_graph(100, m2, seed)
        DBA2 = nx.dual_barabasi_albert_graph(100, m1, m2, 0, seed)
        assert BA2.edges() == DBA2.edges()
        BA3 = nx.barabasi_albert_graph(100, m1, seed)
        DBA3 = nx.dual_barabasi_albert_graph(100, m1, m1, p, seed)
        assert BA3.size() == DBA3.size()
        DBA = nx.dual_barabasi_albert_graph(100, m1, m2, p, seed, initial_graph)
        BA1 = nx.barabasi_albert_graph(100, m1, seed, initial_graph)
        BA2 = nx.barabasi_albert_graph(100, m2, seed, initial_graph)
        assert min(BA1.size(), BA2.size()) <= DBA.size() <= max(BA1.size(), BA2.size())
    dbag = nx.dual_barabasi_albert_graph
    pytest.raises(nx.NetworkXError, dbag, m1, m1, m2, 0)
    pytest.raises(nx.NetworkXError, dbag, m2, m1, m2, 0)
    pytest.raises(nx.NetworkXError, dbag, 100, m1, m2, -0.5)
    pytest.raises(nx.NetworkXError, dbag, 100, m1, m2, 1.5)
    initial = nx.complete_graph(max(m1, m2) - 1)
    pytest.raises(nx.NetworkXError, dbag, 100, m1, m2, p, initial_graph=initial)