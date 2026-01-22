import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def other_phase_species_idxs(self, phase_idx=0):
    return [idx for idx, s in enumerate(self.substances.values()) if s.phase_idx != phase_idx]