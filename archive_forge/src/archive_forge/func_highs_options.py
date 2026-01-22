import logging
from typing import List, Dict, Optional
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import PyomoException
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigValue, NonNegativeInt
from pyomo.common.tee import TeeStream, capture_output
from pyomo.common.log import LogStream
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base import SymbolMap
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.param import _ParamData
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.contrib.appsi.base import (
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
from pyomo.common.dependencies import numpy as np
from pyomo.core.staleflag import StaleFlagManager
import sys
@highs_options.setter
def highs_options(self, val: Dict):
    self._solver_options = val