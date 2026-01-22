from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
@staticmethod
def fromManifoldAndShapes(manifold, shapes, one_cocycle=None):
    if not one_cocycle:
        for cusp_info in manifold.cusp_info():
            if not cusp_info['complete?']:
                raise IncompleteCuspError(manifold)
    if not manifold.is_orientable():
        raise ValueError('Non-orientable')
    m = t3m.Mcomplex(manifold)
    t = TransferKernelStructuresEngine(m, manifold)
    t.reindex_cusps_and_transfer_peripheral_curves()
    t.add_shapes(shapes)
    if one_cocycle == 'develop':
        resolved_one_cocycle = None
    else:
        resolved_one_cocycle = one_cocycle
    c = ComplexCuspCrossSection(m)
    c.add_structures(resolved_one_cocycle)
    c.manifold = manifold
    return c