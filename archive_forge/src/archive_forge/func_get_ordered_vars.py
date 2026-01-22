from typing import List
from pyomo.core.base.param import _ParamData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.block import _BlockData
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.numvalue import value
from pyomo.contrib.appsi.base import PersistentBase
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.kernel.objective import minimize
from .config import WriterConfig
from pyomo.common.collections import OrderedSet
import os
from ..cmodel import cmodel, cmodel_available
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env
def get_ordered_vars(self):
    return [self._solver_var_to_pyomo_var_map[i] for i in self._writer.get_solve_vars()]