from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def safe_log(z):
    l = (branch_factor * z ** N).log()
    if l.imag().abs() > PiMinusEpsilon:
        raise LogToCloseToBranchCutError()
    return l / N