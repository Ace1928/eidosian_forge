import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
def test_viewtype(self):
    nv = self.G.nodes
    ndvfalse = nv.data(False)
    assert nv is ndvfalse
    assert nv is not self.ndv