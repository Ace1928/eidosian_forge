from .line import R13LineWithMatrix
from ..verify.shapes import compute_hyperbolic_shapes # type: ignore
from ..snap.fundamental_polyhedron import FundamentalPolyhedronEngine # type: ignore
from ..snap.kernel_structures import TransferKernelStructuresEngine # type: ignore
from ..snap.t3mlite import simplex, Mcomplex, Tetrahedron, Vertex # type: ignore
from ..SnapPy import word_as_list # type: ignore
from ..hyperboloid import (o13_inverse,  # type: ignore
from ..upper_halfspace import sl2c_inverse, psl2c_to_o13 # type: ignore
from ..upper_halfspace.ideal_point import ideal_point_to_r13 # type: ignore
from ..matrix import vector, matrix, mat_solve # type: ignore
from ..math_basics import prod, xgcd # type: ignore
from collections import deque
from typing import Tuple, Sequence, Optional, Any
def compute_r13_planes_for_tet(tet: Tetrahedron):
    """
    Computes outward facing normals/plane equations from the vertices of
    positively oriented tetrahedra - all in the hyperboloid model.
    """
    tet.R13_unnormalised_planes = {f: unnormalised_plane_eqn_from_r13_points([tet.R13_vertices[v] for v in verts]) for f, verts in simplex.VerticesOfFaceCounterclockwise.items()}
    tet.R13_planes = {f: space_r13_normalise(plane) for f, plane in tet.R13_unnormalised_planes.items()}