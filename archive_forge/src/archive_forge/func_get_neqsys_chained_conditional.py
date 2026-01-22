import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def get_neqsys_chained_conditional(self, rref_equil=False, rref_preserv=False, NumSys=NumSysLin, **kwargs):
    from pyneqsys import ConditionalNeqSys, ChainedNeqSys

    def mk_factory(NS):

        def factory(conds):
            return self._SymbolicSys_from_NumSys(NS, conds, rref_equil, rref_preserv, **kwargs)
        return factory
    return ChainedNeqSys([ConditionalNeqSys([(self._fw_cond_factory(ri), self._bw_cond_factory(ri, NS.small)) for ri in self.phase_transfer_reaction_idxs()], mk_factory(NS)) for NS in NumSys])