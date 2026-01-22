import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
def check_triangulation(embedding, expected_embedding):
    res_embedding, _ = triangulate_embedding(embedding, True)
    assert res_embedding.get_data() == expected_embedding, 'Expected embedding incorrect'
    res_embedding, _ = triangulate_embedding(embedding, False)
    assert res_embedding.get_data() == expected_embedding, 'Expected embedding incorrect'