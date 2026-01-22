import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def nmatch(n1, n2):
    return n1 == n2