import pytest
import networkx as nx
from networkx.algorithms.planarity import (
def test_connect_components(self):
    embedding = nx.PlanarEmbedding()
    embedding.connect_components(1, 2)