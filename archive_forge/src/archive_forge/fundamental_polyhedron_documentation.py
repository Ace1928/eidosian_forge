from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage

        Given the result of M.fundamental_group(...) where M is the
        corresponding SnapPy.Manifold, return the matrices for that
        presentation of the fundamental polyhedron.

        The GeneratorMatrices computed here are for the face-pairing
        presentation with respect to the fundamental polyhedron.
        That presentation can be simplfied by M.fundamental_group(...)
        and this function will compute the matrices for the simplified
        presentation from the GeneratorMatrices.

        If match_kernel is True, it will flip the signs of some of
        the matrices to match the ones in the given G (which were determined
        by the SnapPea kernel).

        This makes the result stable when changing precision (when normalizing
        matrices with determinant -1, sqrt(-1) might jump between i and -i when
        increasing precision).
        