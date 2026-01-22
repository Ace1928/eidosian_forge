from . import exceptions
from . import epsilons
from . import debug
from .tracing import trace_geodesic
from .crush import crush_geodesic_pieces
from .line import R13LineWithMatrix
from .geometric_structure import add_r13_geometry, word_to_psl2c_matrix
from .geodesic_info import GeodesicInfo, sample_line
from .perturb import perturb_geodesics
from .subdivide import traverse_geodesics_to_subdivide
from .cusps import (
from ..snap.t3mlite import Mcomplex
from ..exceptions import InsufficientPrecisionError
import functools
from typing import Sequence
def drill_words(manifold, words: Sequence[str], verified: bool=False, bits_prec=None, verbose: bool=False):
    """
    A generalization of M.drill_word taking a list of words to
    drill several geodesics simultaneously.

    Here is an example where we drill the core curve corresponding to the third cusp
    and a geodesic that is not a core curve:


        >>> from snappy import Manifold
        >>> M=Manifold("t12047(0,0)(1,3)(1,4)(1,5)")
        >>> [ info.get('core_length') for info in M.cusp_info() ] # doctest: +NUMERIC9
        [None,
         0.510804267610103 + 1.92397456664239*I,
         0.317363079597924 + 1.48157893409218*I,
         0.223574975263386 + 1.26933288854145*I]
        >>> G = M.fundamental_group(simplify_presentation = False)
        >>> G.complex_length('c') # doctest: +NUMERIC9
        0.317363079597924 + 1.48157893409218*I
        >>> G.complex_length('fA') # doctest: +NUMERIC9
        1.43914411734250 + 2.66246879992795*I
        >>> N = M.drill_words(['c','fA'])
        >>> N
        t12047_drilled(0,0)(1,3)(1,5)(0,0)(0,0)

    The last n cusps correspond to the n geodesics that were drilled, appearing
    in the same order the words for the geodesics were given. Note that in the
    above example, the drilled manifold has only five cusps even though the
    original manifold had four cusps and we drilled two geodesics. This is
    because one geodesic was a core curve. The corresponding cusp was unfilled
    (from (1,4)) and grouped with the other cusps coming from drilling.

    We obtain the original (undrilled) manifold by (1,0)-filling the last n cusps.

        >>> N.dehn_fill((1,0), 3)
        >>> N.dehn_fill((1,0), 4)
        >>> M.is_isometric_to(N)
        True
        >>> [ info.get('core_length') for info in N.cusp_info() ] # doctest: +NUMERIC9
        [None,
         0.510804267610103 + 1.92397456664239*I,
         0.223574975263386 + 1.26933288854145*I,
         0.317363079597924 + 1.48157893409218*I,
         1.43914411734251 + 2.66246879992796*I]

    """
    if isinstance(words, str):
        raise ValueError('words has to be a list of strings, not a single string.')
    if len(words) == 0:
        return manifold.copy()
    if not manifold.is_orientable():
        raise ValueError('Drilling only supported for orientable manifolds.')
    try:
        return drill_words_implementation(manifold, words=words, verified=verified, bits_prec=bits_prec, verbose=verbose)
    except exceptions.GeodesicHittingOneSkeletonError:
        pass
    try:
        return drill_words_implementation(manifold, words=words, verified=verified, bits_prec=bits_prec, perturb=True, verbose=verbose)
    except exceptions.RayHittingOneSkeletonError as e:
        raise InsufficientPrecisionError('The geodesic is so closer to an edge of the triangulation that it cannot be unambiguously traced with the current precision. Increasing the precision should solve this problem.') from e