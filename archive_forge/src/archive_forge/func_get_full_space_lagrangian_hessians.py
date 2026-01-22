import itertools
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.timing import HierarchicalTimer
from pyomo.util.subsystems import create_subsystem_block
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
import numpy as np
import scipy.sparse as sps
def get_full_space_lagrangian_hessians(self):
    """
        Calculates terms of Hessian of full-space Lagrangian due to
        external and residual constraints. Note that multipliers are
        set by set_equality_constraint_multipliers. These matrices
        are used to calculate the Hessian of the reduced-space
        Lagrangian.

        """
    nlp = self._nlp
    x = self.input_vars
    y = self.external_vars
    hlxx = nlp.extract_submatrix_hessian_lag(x, x)
    hlxy = nlp.extract_submatrix_hessian_lag(x, y)
    hlyy = nlp.extract_submatrix_hessian_lag(y, y)
    return (hlxx, hlxy, hlyy)