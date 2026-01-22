import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_main_crust(self):
    main_crust_subgraph = nx.k_crust(self.H)
    assert sorted(main_crust_subgraph.nodes()) == [0, 1, 3]