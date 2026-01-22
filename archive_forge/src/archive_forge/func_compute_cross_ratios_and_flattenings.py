from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def compute_cross_ratios_and_flattenings(tet, index):

    def get_ptolemy_coordinate(addl_index):
        total_index = matrix.vector_add(index, addl_index)
        key = 'c_%d%d%d%d' % tuple(total_index) + '_%d' % tet
        return solution_dict[key]

    def get_obstruction_variable(face):
        key = 's_%d_%d' % (face, tet)
        return solution_dict[key]
    c1010 = get_ptolemy_coordinate((1, 0, 1, 0))
    c1001 = get_ptolemy_coordinate((1, 0, 0, 1))
    c0110 = get_ptolemy_coordinate((0, 1, 1, 0))
    c0101 = get_ptolemy_coordinate((0, 1, 0, 1))
    c1100 = get_ptolemy_coordinate((1, 1, 0, 0))
    c0011 = get_ptolemy_coordinate((0, 0, 1, 1))
    z = c1010 * c0101 / (c1001 * c0110)
    zp = -(c1001 * c0110) / (c1100 * c0011)
    zpp = c1100 * c0011 / (c1010 * c0101)
    if has_obstruction:
        s0 = get_obstruction_variable(0)
        s1 = get_obstruction_variable(1)
        s2 = get_obstruction_variable(2)
        s3 = get_obstruction_variable(3)
        z = s0 * s1 * z
        zp = s0 * s2 * zp
        zpp = s0 * s3 * zpp
    variable_end = '_%d%d%d%d' % tuple(index) + '_%d' % tet
    if as_flattenings:

        def make_triple(w, z):
            z = _convert_to_pari_float(z)
            return (w, z, ((w - z.log()) / f).round())
        w = _compute_flattening(c1010, c0101, c1001, c0110, branch_factor, evenN)
        wp = _compute_flattening(c1001, c0110, c1100, c0011, branch_factor, evenN)
        wpp = _compute_flattening(c1100, c0011, c1010, c0101, branch_factor, evenN)
        return [('z' + variable_end, make_triple(w, z)), ('zp' + variable_end, make_triple(wp, zp)), ('zpp' + variable_end, make_triple(wpp, zpp))]
    else:
        return [('z' + variable_end, z), ('zp' + variable_end, zp), ('zpp' + variable_end, zpp)]