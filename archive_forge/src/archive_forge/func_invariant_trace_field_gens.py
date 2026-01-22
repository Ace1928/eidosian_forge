from .shapes import polished_tetrahedra_shapes
from ..sage_helper import _within_sage, sage_method
from .polished_reps import polished_holonomy
from . import nsagetools, interval_reps, slice_obs_HKL
from .character_varieties import character_variety, character_variety_ideal
@sage_method
def invariant_trace_field_gens(manifold, fundamental_group_args=[]):
    """
    The generators of the trace field as ApproximateAlgebraicNumbers. Can be
    used to compute the tetrahedra field, where the first two parameters
    are bits of precision and maximum degree of the field::

        sage: M = Manifold('m007(3,1)')
        sage: K = M.invariant_trace_field_gens().find_field(100, 10, optimize=True)[0]
        sage: L = M.trace_field_gens().find_field(100, 10, optimize=True)[0]
        sage: K.polynomial(), L.polynomial()
        (x^2 - x + 1, x^4 - 2*x^3 + x^2 + 6*x + 3)
    """

    def func(prec):
        return polished_holonomy(manifold, prec, fundamental_group_args).invariant_trace_field_generators()
    return ListOfApproximateAlgebraicNumbers(func)