import pytest
import networkx as nx
@pytest.mark.parametrize('generator', _gnp_generators)
def test_gnp_generators_for_p_close_to_1(generator):
    """If the edge probability `p` is close to 1, the resulting graph should have all edges."""
    runs = 100
    edges = sum((generator(10, 0.99999, directed=True).number_of_edges() for _ in range(runs)))
    assert abs(edges / float(runs) - 90) <= runs * 2.0 / 100