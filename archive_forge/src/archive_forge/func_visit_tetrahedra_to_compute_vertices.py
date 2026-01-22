from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def visit_tetrahedra_to_compute_vertices(self, init_tet, init_vertices):
    """
        Computes the positions of the vertices of fundamental polyhedron in
        the boundary of H^3, assuming the Mcomplex has been unglued and
        ShapeParameters were assigned to the tetrahedra.

        It starts by assigning the vertices of the given init_tet using
        init_vertices.
        """
    for vertex in self.mcomplex.Vertices:
        vertex.IdealPoint = None
    for tet in self.mcomplex.Tetrahedra:
        tet.visited = False
    self.mcomplex.InitialTet = init_tet
    for v, idealPoint in init_vertices.items():
        init_tet.Class[v].IdealPoint = idealPoint
    init_tet.visited = True
    queue = [init_tet]
    while len(queue) > 0:
        tet = queue.pop(0)
        for F in simplex.TwoSubsimplices:
            if bool(tet.Neighbor[F]) != bool(tet.GeneratorsInfo[F] == 0):
                raise Exception('Improper fundamental domain, probably a bug in unglue code')
            S = tet.Neighbor[F]
            if S and (not S.visited):
                perm = tet.Gluing[F]
                for V in _VerticesInFace[F]:
                    vertex_class = S.Class[perm.image(V)]
                    if vertex_class.IdealPoint is None:
                        vertex_class.IdealPoint = tet.Class[V].IdealPoint
                _compute_fourth_corner(S)
                S.visited = True
                queue.append(S)