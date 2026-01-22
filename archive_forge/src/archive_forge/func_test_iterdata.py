import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
def test_iterdata(self):
    G = self.G.copy()
    evr = self.eview(G)
    ev = evr(data=True)
    ev_def = evr(data='foo', default=1)
    for u, v, d in ev:
        pass
    assert d == {}
    for u, v, wt in ev_def:
        pass
    assert wt == 1
    self.modify_edge(G, (2, 3), foo='bar')
    for e in ev:
        assert len(e) == 3
        if set(e[:2]) == {2, 3}:
            assert e[2] == {'foo': 'bar'}
            checked = True
        else:
            assert e[2] == {}
    assert checked
    for e in ev_def:
        assert len(e) == 3
        if set(e[:2]) == {2, 3}:
            assert e[2] == 'bar'
            checked_wt = True
        else:
            assert e[2] == 1
    assert checked_wt