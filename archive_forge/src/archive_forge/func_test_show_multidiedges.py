import pytest
import networkx as nx
def test_show_multidiedges(self):
    factory = nx.classes.filters.show_multidiedges
    f = factory([(1, 2, 0), (3, 4, 1), (1, 2, 1)])
    assert f(1, 2, 0)
    assert f(1, 2, 1)
    assert not f(1, 2, 2)
    assert not f(3, 4, 0)
    assert f(3, 4, 1)
    assert not f(4, 3, 1)
    assert not f(4, 3, 0)
    assert not f(2, 3, 0)
    assert not f(0, -1, 0)
    assert not f('a', 'b', 0)
    pytest.raises(TypeError, f, 1, 2, 3, 4)
    pytest.raises(TypeError, f, 1, 2)
    pytest.raises(TypeError, f, 1)
    pytest.raises(TypeError, f)
    pytest.raises(TypeError, factory, [1, 2, 3])
    pytest.raises(ValueError, factory, [(1, 2)])
    pytest.raises(ValueError, factory, [(1, 2, 3, 4)])