import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
def n_its(self, nodes):
    return {(node, 'bar' if node == 3 else 1) for node in nodes}