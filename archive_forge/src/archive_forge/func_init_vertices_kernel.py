from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def init_vertices_kernel(self):
    """
        Computes vertices for the initial tetrahedron matching the choices
        made by the SnapPea kernel.
        """
    tet = self.mcomplex.ChooseGenInitialTet
    candidates = []
    for perm in Perm4.A4():
        z = tet.ShapeParameters[perm.image(simplex.E01)]
        CF = z.parent()
        sqrt_z = z.sqrt()
        sqrt_z_inv = CF(1) / sqrt_z
        candidate = {perm.image(simplex.V0): Infinity, perm.image(simplex.V1): CF(0), perm.image(simplex.V2): sqrt_z_inv, perm.image(simplex.V3): sqrt_z}
        if _are_vertices_close_to_kernel(candidate, tet.SnapPeaIdealVertices):
            candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0]
    raise Exception('Could not match vertices to vertices from SnapPea kernel')