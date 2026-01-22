import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def non_precip_rids(self, precipitates):
    return [idx for idx, precip in zip(self.phase_transfer_reaction_idxs(), precipitates) if not precip]