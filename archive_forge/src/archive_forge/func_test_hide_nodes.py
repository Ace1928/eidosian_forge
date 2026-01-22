import pytest
import networkx as nx
def test_hide_nodes(self):
    f = nx.classes.filters.hide_nodes([1, 2, 3])
    assert not f(1)
    assert not f(2)
    assert not f(3)
    assert f(4)
    assert f(0)
    assert f('a')
    pytest.raises(TypeError, f, 1, 2)
    pytest.raises(TypeError, f)