from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def unmarked_arc(self, vertex):
    """
        Starting at this vertex, find an unmarked edge and follow its
        component until we run into a vertex with at least one marked
        edge.  Remove loops to get an embedded arc. Return the list
        of edges traversed by the embedded arc.
        """
    valence = self.marked_valence(vertex)
    if valence == 4:
        raise ValueError('Vertex must have unmarked edges.')
    if valence == 0:
        raise ValueError('Vertex must be in the marked subgraph.')
    edges, vertices, seen = ([], [], set())
    for first_edge in self(vertex):
        if not first_edge.marked:
            break
    for edge in self.path(vertex, first_edge):
        edges.append(edge)
        vertex = edge(vertex)
        if self.marked_valence(vertex) > 0:
            break
        if vertex in seen:
            n = vertices.index(vertex)
            edges = edges[:n + 1]
            vertices = vertices[:n + 1]
            seen = set(vertices)
        else:
            vertices.append(vertex)
            seen.add(vertex)
    return (edges, vertex)