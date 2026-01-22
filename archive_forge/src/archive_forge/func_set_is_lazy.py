from sys import version_info as _swig_python_version_info
import numbers
from ortools.linear_solver.python.linear_solver_natural_api import OFFSET_KEY
from ortools.linear_solver.python.linear_solver_natural_api import inf
from ortools.linear_solver.python.linear_solver_natural_api import LinearExpr
from ortools.linear_solver.python.linear_solver_natural_api import ProductCst
from ortools.linear_solver.python.linear_solver_natural_api import Sum
from ortools.linear_solver.python.linear_solver_natural_api import SumArray
from ortools.linear_solver.python.linear_solver_natural_api import SumCst
from ortools.linear_solver.python.linear_solver_natural_api import LinearConstraint
from ortools.linear_solver.python.linear_solver_natural_api import VariableExpr
def set_is_lazy(self, laziness):
    """
        Advanced usage: sets the constraint "laziness".

        **This is only supported for SCIP and has no effect on other
        solvers.**

        When **laziness** is true, the constraint is only considered by the Linear
        Programming solver if its current solution violates the constraint. In this
        case, the constraint is definitively added to the problem. This may be
        useful in some MIP problems, and may have a dramatic impact on performance.

        For more info see: http://tinyurl.com/lazy-constraints.
        """
    return _pywraplp.Constraint_set_is_lazy(self, laziness)