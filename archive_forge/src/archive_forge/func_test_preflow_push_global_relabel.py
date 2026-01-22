import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_preflow_push_global_relabel(self):
    G = read_graph('gw1')
    R = preflow_push(G, 1, len(G), global_relabel_freq=50)
    assert R.graph['flow_value'] == 1202018