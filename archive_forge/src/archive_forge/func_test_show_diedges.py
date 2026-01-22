import pytest
import networkx as nx
def test_show_diedges(self):
    factory = nx.classes.filters.show_diedges
    f = factory([(1, 2), (3, 4)])
    assert f(1, 2)
    assert f(3, 4)
    assert not f(4, 3)
    assert not f(2, 3)
    assert not f(0, -1)
    assert not f('a', 'b')
    pytest.raises(TypeError, f, 1, 2, 3)
    pytest.raises(TypeError, f, 1)
    pytest.raises(TypeError, f)
    pytest.raises(TypeError, factory, [1, 2, 3])
    pytest.raises(ValueError, factory, [(1, 2, 3)])