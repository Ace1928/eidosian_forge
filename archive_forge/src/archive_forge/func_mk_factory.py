import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def mk_factory(NS):

    def factory(conds):
        return self._SymbolicSys_from_NumSys(NS, conds, rref_equil, rref_preserv, **kwargs)
    return factory