from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def flattening_condition(r):
    return 3 * r * [0] + 3 * [1] + 3 * (num_tets - r - 1) * [0]