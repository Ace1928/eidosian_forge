import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def same_component(u, v):
    return component_of[u] == component_of[v]