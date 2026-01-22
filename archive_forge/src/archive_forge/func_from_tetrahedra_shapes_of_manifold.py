from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
@classmethod
def from_tetrahedra_shapes_of_manifold(cls, M):
    """
        Takes as argument a manifold and produces (weak) flattenings using
        the tetrahedra_shapes of the manifold M.

        >>> from snappy import Manifold
        >>> M = Manifold("5_2")
        >>> flattenings = Flattenings.from_tetrahedra_shapes_of_manifold(M)
        >>> flattenings.check_against_manifold(M)
        >>> flattenings.check_against_manifold()
        """
    PiI = pari('Pi * I')
    num_tets = M.num_tetrahedra()
    z_cross_ratios = M.tetrahedra_shapes(part='rect', dec_prec=pari.get_real_precision())
    all_cross_ratios = sum([[z, 1 / (1 - z), 1 - 1 / z] for z in z_cross_ratios], [])
    log_all_cross_ratios = [z.log() for z in all_cross_ratios]

    def flattening_condition(r):
        return 3 * r * [0] + 3 * [1] + 3 * (num_tets - r - 1) * [0]
    flattening_conditions = [flattening_condition(r) for r in range(num_tets)]
    try:
        equations = M.gluing_equations().data
    except AttributeError:
        equations = [[int(c) for c in row] for row in M.gluing_equations().rows()]
    all_equations = equations + flattening_conditions
    u, v, d_mat = matrix.smith_normal_form(all_equations)
    extra_cols = len(all_equations[0]) - len(all_equations)
    d = [d_mat[r][r + extra_cols] for r in range(len(d_mat))]
    errors = matrix.matrix_mult_vector(all_equations, log_all_cross_ratios)
    int_errors = [(x / PiI).real().round() for x in errors]
    int_errors_in_other_basis = matrix.matrix_mult_vector(u, int_errors)

    def quotient(x, y):
        if x == 0 and y == 0:
            return 0
        assert x % y == 0, '%s %s' % (x, y)
        return x / y
    flattenings_in_other_basis = extra_cols * [0] + [-quotient(x, y) for x, y in zip(int_errors_in_other_basis, d)]
    flattenings = matrix.matrix_mult_vector(v, flattenings_in_other_basis)
    assert matrix.matrix_mult_vector(all_equations, flattenings) == [-x for x in int_errors]
    keys = sum([['z_0000_%d' % i, 'zp_0000_%d' % i, 'zpp_0000_%d' % i] for i in range(num_tets)], [])
    Mcopy = M.copy()
    return Flattenings(dict([(k, (log + PiI * p, z, p)) for k, log, z, p in zip(keys, log_all_cross_ratios, all_cross_ratios, flattenings)]), manifold_thunk=lambda: Mcopy)