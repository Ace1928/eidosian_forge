import pytest
from networkx.exception import NetworkXError
from networkx.generators.duplication import (
def test_probability_too_small(self):
    with pytest.raises(NetworkXError):
        duplication_divergence_graph(3, -1)