import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.parametrize('non_mc', ('0123', ['0', '1', '2', '3']))
def test_relabel_nodes_non_mapping_or_callable(self, non_mc):
    """If `mapping` is neither a Callable or a Mapping, an exception
        should be raised."""
    G = nx.path_graph(4)
    with pytest.raises(AttributeError):
        nx.relabel_nodes(G, non_mc)