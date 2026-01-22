import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding
def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)