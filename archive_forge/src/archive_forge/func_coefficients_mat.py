import abc
import math
import functools
from numbers import Integral
from collections.abc import Iterable, MutableSequence
from enum import Enum
from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel, Objective, maximize, minimize, Block
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.base.var import Var, IndexedVar
from pyomo.core.expr.numvalue import value, native_numeric_types
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import add_bounds_for_uncertain_parameters
@property
def coefficients_mat(self):
    """
        (L + N, N) numpy.ndarray : Coefficient matrix of all polyhedral
        constraints defining the budget set. Composed from the incidence
        matrix used for defining the budget constraints and a
        coefficient matrix for individual uncertain parameter
        nonnegativity constraints.

        This attribute cannot be set. The budget constraint
        incidence matrix may be altered through the
        `budget_membership_mat` attribute.
        """
    return np.append(self.budget_membership_mat, -np.identity(self.dim), axis=0)