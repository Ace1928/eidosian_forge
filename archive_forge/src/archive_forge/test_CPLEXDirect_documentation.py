import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus
from pyomo.solvers.plugins.solvers.cplex_direct import (
Test that:
        - The variable is added correctly to `solver_model`
        - The CPLEX `variables` interface is called only once
        - Fixed variable bounds are set correctly
        