import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
def test_multiple_component_graph1():
    embedding_data = {0: [], 1: []}
    check_embedding_data(embedding_data)