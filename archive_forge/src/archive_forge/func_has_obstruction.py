from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def has_obstruction(self):
    """
        Whether the Ptolemy variety has legacy obstruction class that
        modifies the Ptolemy relation to
        """
    N, has_obstruction = _N_and_has_obstruction_for_ptolemys(self)
    return has_obstruction