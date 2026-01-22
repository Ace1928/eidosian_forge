import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def assert_components_equal(x, y):
    sx = {frozenset(c) for c in x}
    sy = {frozenset(c) for c in y}
    assert sx == sy