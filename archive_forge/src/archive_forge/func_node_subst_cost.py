import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def node_subst_cost(uattr, vattr):
    if uattr['color'] == vattr['color']:
        return 1
    else:
        return 10