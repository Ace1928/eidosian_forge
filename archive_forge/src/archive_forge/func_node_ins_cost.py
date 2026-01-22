import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def node_ins_cost(attr):
    if attr['color'] == 'blue':
        return 40
    else:
        return 100