import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def ematch(e1, e2):
    return e1 == e2