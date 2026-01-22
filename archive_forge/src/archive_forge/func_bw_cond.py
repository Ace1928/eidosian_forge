import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def bw_cond(x, p):
    precipitate_idx = rxn.precipitate_stoich(self.substances)[2]
    if x[precipitate_idx] < small:
        return False
    else:
        return True