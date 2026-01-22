import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
@property
@deprecated(last_supported_version='0.3.1', will_be_missing_in='0.8.0', use_instead=phase_transfer_reaction_idxs)
def precipitate_rxn_idxs(self):
    return [idx for idx, rxn in enumerate(self.rxns) if rxn.has_precipitates(self.substances)]