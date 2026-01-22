import pytest
import networkx as nx
from networkx.utils import pairwise
def weight_f(u, v, d):
    return d.get(weight, 1)