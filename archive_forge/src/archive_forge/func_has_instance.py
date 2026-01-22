from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.core.base.block import _BlockData
from pyomo.core.kernel.block import IBlock
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.kernel.suffix import import_suffix_generator
from pyomo.core.expr.numvalue import native_numeric_types, value
from pyomo.core.expr.visitor import evaluate_expression
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.sos import SOSConstraint
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
import time
import logging
def has_instance(self):
    """
        True if set_instance has been called and this solver interface has a pyomo model and a solver model.

        Returns
        -------
        tmp: bool
        """
    return self._pyomo_model is not None