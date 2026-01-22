import pytest
import networkx as nx
@pytest.mark.parametrize('generator', _gnp_generators)
@pytest.mark.parametrize('directed', (True, False))
def test_gnp_generators_basic(generator, directed):
    """If the edge probability `p` is >0 and <1, test only the basic properties."""
    G = generator(10, 0.1, directed=directed)
    assert len(G) == 10
    assert G.is_directed() == directed