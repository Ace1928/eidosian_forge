import random
from .exceptions import ExteriorToLinkError
from .simplify_to_base_tri import good_simplification
from . import put_in_S3
from . import link_projection
from .rational_linear_algebra import Matrix
from . import hyp_utils
from ..SnapPy import set_rand_seed
def filled_is_3sphere(manifold):
    """
    >>> isosig = 'nLvLLLPQQkcfejimhklkmlkmuphkvuoupilhhv_Bbba(1, 0)'
    >>> filled_is_3sphere(Manifold(isosig))
    True
    >>> filled_is_3sphere(Triangulation('m004(1, 2)'))
    False
    """
    if hasattr(manifold, 'without_hyperbolic_structure'):
        T = manifold.without_hyperbolic_structure()
    else:
        T = manifold.copy()
    for i in range(T.num_cusps()):
        if T.cusp_info(i).is_complete:
            T.dehn_fill((1, 0), i)
        for i in range(10):
            if T.fundamental_group().num_generators() == 0:
                return True
            F = T.filled_triangulation()
            if F.fundamental_group().num_generators() == 0:
                return True
            T.randomize()
    return False