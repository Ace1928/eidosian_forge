import pytest
import networkx as nx
from networkx.classes import Graph, MultiDiGraph
from networkx.generators.directed import (
def test_smoke_test_random_graphs(self):
    gn_graph(100)
    gnr_graph(100, 0.5)
    gnc_graph(100)
    scale_free_graph(100)
    gn_graph(100, seed=42)
    gnr_graph(100, 0.5, seed=42)
    gnc_graph(100, seed=42)
    scale_free_graph(100, seed=42)