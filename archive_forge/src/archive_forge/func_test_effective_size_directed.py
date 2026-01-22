import math
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
def test_effective_size_directed(self):
    effective_size = nx.effective_size(self.D)
    assert effective_size[0] == pytest.approx(1.167, abs=0.001)
    assert effective_size[1] == pytest.approx(1.167, abs=0.001)
    assert effective_size[2] == pytest.approx(1, abs=0.001)