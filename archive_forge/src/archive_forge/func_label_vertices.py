import networkx as nx
from .. import t3mlite as t3m
from ..t3mlite.simplex import *
from . import surface
def label_vertices(self):
    N = self.parent_triangulation
    for vert in self.vertices:
        vert.index = None
    for edge in N.Edges:
        corner = edge.Corners[0]
        tet = corner.Tetrahedron
        a = Head[corner.Subsimplex]
        b = Tail[corner.Subsimplex]
        sign = edge.orientation_with_respect_to(tet, a, b)
        for v, s in [(a, sign), (b, -sign)]:
            label = s * (edge.Index + 1)
            i = TruncatedSimplexCorners[v].index(a | b)
            tri = tet.CuspCorners[v]
            tri.vertices[i].index = label