import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def edge_del_cost(attr):
    if attr['color'] == 'blue':
        return 0.2
    else:
        return 0.5