import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
def test_weight(self):
    dv = self.dview(self.G)
    dvw = dv(0, weight='foo')
    assert dvw == 0
    dvw = dv(1, weight='foo')
    assert dvw == 1
    dvw = dv([2, 3], weight='foo')
    assert sorted(dvw) == [(2, 1), (3, 6)]
    dvd = dict(dv(weight='foo'))
    assert dvd[0] == 0
    assert dvd[1] == 1
    assert dvd[2] == 1
    assert dvd[3] == 6