import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
def test_three_node_graph():
    embedding_data = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    check_embedding_data(embedding_data)