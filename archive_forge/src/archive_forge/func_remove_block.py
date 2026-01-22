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
def remove_block(self, block):
    """Remove a single block from the solver's model.

        This will keep any other model components intact.

        WARNING: Users must call remove_block BEFORE modifying the block.

        Parameters
        ----------
        block: Block (scalar Block or a single _BlockData)

        """
    for sub_block in block.block_data_objects(descend_into=True, active=True):
        for con in sub_block.component_data_objects(ctype=Constraint, descend_into=False, active=True):
            self.remove_constraint(con)
        for con in sub_block.component_data_objects(ctype=SOSConstraint, descend_into=False, active=True):
            self.remove_sos_constraint(con)
    for var in block.component_data_objects(ctype=Var, descend_into=True, active=True):
        self.remove_var(var)