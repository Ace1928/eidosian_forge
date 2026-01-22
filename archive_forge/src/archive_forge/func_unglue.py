from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def unglue(self):
    """
        It will unglue all face-pairings corresponding to generators.
        What is left is a fundamental polyhedron.

        It assumes that GeneratorsInfo has been set (by the
        TranferKernelStructuresEngine).

        Besides ungluing, it will add the field Generators to the Mcomplex
        and SubsimplexIndexInManifold to each Vertex, Edge, Face, see
        examples in from_manifold_and_shapes.
        """
    originalSubsimplexIndices = [[tet.Class[subsimplex].Index for subsimplex in range(1, 15)] for tet in self.mcomplex.Tetrahedra]
    self.mcomplex.Generators = {}
    for tet in self.mcomplex.Tetrahedra:
        for face in simplex.TwoSubsimplices:
            g = tet.GeneratorsInfo[face]
            if g != 0:
                l = self.mcomplex.Generators.setdefault(g, [])
                l.append(((Corner(tet, face), Corner(tet.Neighbor[face], tet.Gluing[face].image(face))), tet.Gluing[face]))
    for g, pairings in self.mcomplex.Generators.items():
        if g > 0:
            for corners, perm in pairings:
                for corner in corners:
                    corner.Tetrahedron.attach(corner.Subsimplex, None, None)
    self.mcomplex.rebuild()
    for tet, o in zip(self.mcomplex.Tetrahedra, originalSubsimplexIndices):
        for subsimplex, index in enumerate(o):
            tet.Class[subsimplex + 1].SubsimplexIndexInManifold = index