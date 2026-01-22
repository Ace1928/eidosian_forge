import pytest
import networkx as nx
from networkx.algorithms.centrality.subgraph_alg import (
def test_communicability_betweenness_centrality(self):
    answer = {0: 0.07017447951484615, 1: 0.7156559870110799, 2: 0.7156559870110799, 3: 0.07017447951484615}
    result = communicability_betweenness_centrality(nx.path_graph(4))
    for k, v in result.items():
        assert answer[k] == pytest.approx(v, abs=1e-07)
    answer1 = {'1': 0.06003907419394952, 'Albert': 0.315470761661372, 'Aric': 0.3154707616613721, 'Dan': 0.682977786783162, 'Franck': 0.21977926617449497}
    G1 = nx.Graph([('Franck', 'Aric'), ('Aric', 'Dan'), ('Dan', 'Albert'), ('Albert', 'Franck'), ('Dan', '1'), ('Franck', 'Albert')])
    result1 = communicability_betweenness_centrality(G1)
    for k, v in result1.items():
        assert answer1[k] == pytest.approx(v, abs=1e-07)