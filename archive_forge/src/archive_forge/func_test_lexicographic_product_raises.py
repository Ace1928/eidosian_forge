import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_lexicographic_product_raises():
    with pytest.raises(nx.NetworkXError):
        P = nx.lexicographic_product(nx.DiGraph(), nx.Graph())