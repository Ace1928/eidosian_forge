import pytest
import networkx as nx
@pytest.mark.parametrize('generator', _gnp_generators)
@pytest.mark.parametrize(('directed', 'expected_num_edges'), [(False, 45), (True, 90)])
def test_gnp_generators_greater_than_1_edge_probability(generator, directed, expected_num_edges):
    """If the edge probability `p` is >=1, the resulting graph should be complete."""
    G = generator(10, 1.1, directed=directed)
    assert len(G) == 10
    assert G.number_of_edges() == expected_num_edges
    assert G.is_directed() == directed