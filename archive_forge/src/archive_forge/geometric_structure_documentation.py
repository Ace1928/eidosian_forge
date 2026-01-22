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

    Given one of the dictionaries returned by Manifold.cusp_info(),
    returns the "filling matrix" filling_matrix.

    filling_matrix is a matrix of integers (as list of lists) such that
    filling_matrix[0] contains the filling coefficients
    (e.g., [3,4] for m004(3,4)) and the determinant is 1 if the cusp is
    filled. That is, filling_matrix[1] determines a curve intersecting
    the filling curve once (as sum of a multiple of meridian and
    longitude) and that is thus parallel to the core curve.

    For an unfilled cusp, filling_matrix is ((0,0), (0,0))

    Raises an exception if the filling coefficients are non-integral or
    not coprime.
    