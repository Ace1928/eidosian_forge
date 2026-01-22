import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
@property
@deprecated(last_supported_version='0.3.1', will_be_missing_in='0.8.0', use_instead=other_phase_species_idxs)
def precipitate_substance_idxs(self):
    return [idx for idx, s in enumerate(self.substances.values()) if s.precipitate]