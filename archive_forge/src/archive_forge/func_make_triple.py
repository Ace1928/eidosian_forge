from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def make_triple(w, z):
    z = _convert_to_pari_float(z)
    return (w, z, ((w - z.log()) / f).round())