from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
@staticmethod
def from_snappy_manifold(M, dec_prec=None, bits_prec=None, intervals=False):
    """
        Constructs an assignment of shape parameters/cross ratios using
        the tetrahehdra_shapes method of a given SnapPy manifold. The optional
        parameters are the same as that of tetrahedra_shapes.
        """
    shapes = M.tetrahedra_shapes('rect', dec_prec=dec_prec, bits_prec=bits_prec, intervals=intervals)
    d = {}
    for i, shape in enumerate(shapes):
        d['z_0000_%d' % i] = shape
        d['zp_0000_%d' % i] = 1 / (1 - shape)
        d['zpp_0000_%d' % i] = 1 - 1 / shape
    return CrossRatios(d, is_numerical=True, manifold_thunk=lambda M=M: M)