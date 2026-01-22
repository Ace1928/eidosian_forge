from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def evaluate_all_for_root(root):

    def evaluate_key_for_root(key, value):
        v = _evaluate_at_root(value, root)
        if key[:2] == 'z_':
            z = v
            zp = 1 / (1 - z)
            zpp = 1 - 1 / z
            return [(key, z), ('zp_' + key[2:], zp), ('zpp_' + key[2:], zpp)]
        elif key[:3] == 'zp_' or key[:4] == 'zpp_':
            return []
        else:
            return [(key, v)]
    return dict(sum([evaluate_key_for_root(key, value) for key, value in d.items()], []))