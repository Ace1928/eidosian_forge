import pytest
import networkx as nx
def test_show_nodes(self):
    f = nx.classes.filters.show_nodes([1, 2, 3])
    assert f(1)
    assert f(2)
    assert f(3)
    assert not f(4)
    assert not f(0)
    assert not f('a')
    pytest.raises(TypeError, f, 1, 2)
    pytest.raises(TypeError, f)