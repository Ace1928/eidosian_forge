import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def substance_labels(self, latex=False):
    if latex:
        result = ['$' + s.latex_name + '$' for s in self.substances.values()]
        return result
    else:
        return [s.name for s in self.substances.values()]