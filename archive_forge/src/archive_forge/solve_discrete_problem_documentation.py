from copy import deepcopy
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.util import (
from pyomo.core import Objective, Constraint
from pyomo.opt import SolutionStatus, SolverFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
Distinguish between an infeasible or unbounded solution.

    Linear solvers will sometimes tell me that a problem is infeasible or
    unbounded during presolve, but not distinguish between the two cases. We
    address this by solving again with a solver option flag on.

    