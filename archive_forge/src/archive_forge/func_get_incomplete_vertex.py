from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def get_incomplete_vertex(self):
    """
        Return a vertex with some marked and some unmarked edges.
        If there are any, return a vertex with marked valence 3.
        """
    G = self.fat_graph
    vertices = [v for v in G.vertices if 0 < G.marked_valences[v] < 4]
    vertices.sort(key=lambda v: G.marked_valences[v])
    try:
        return vertices.pop()
    except IndexError:
        return None