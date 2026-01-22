import pyomo.environ as pyo
from pyomo.opt import check_optimal_termination
from pyomo.common.dependencies import attempt_import
from .interface import InteriorPointInterface
from .linalg.scipy_interface import ScipyInterface

    This function computes the inverse of the reduced Hessian of a problem at the
    solution. This function first solves the problem with Ipopt and then generates
    the KKT system for the barrier subproblem to compute the inverse reduced hessian.

    For more information on the reduced Hessian, see "Numerical Optimization", 2nd Edition
    Nocedal and Wright, 2006.

    The approach used in this method can be found in, "Computational Strategies for
    the Optimal Operation of Large-Scale Chemical Processes", Dissertation, V. Zavala
    2008. See section 3.2.1.

    Parameters
    ----------
    model : Pyomo model
        The Pyomo model that we want to solve and analyze
    independent_variables : list of Pyomo variables
        This is the list of independent variables for computing the reduced hessian.
        These variables must not be at their bounds at the solution of the
        optimization problem.
    bound_tolerance : float
       The tolerance to use when checking if the variables are too close to their bound.
       If they are too close, then the routine will exit without a reduced hessian.
    solver_options: dictionary
        Additional solver options to consider.
    tee : bool
       This flag is sent to the tee option of the solver. If true, then the solver
       log is output to the console.
    