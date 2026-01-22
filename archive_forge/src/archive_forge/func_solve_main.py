import math
from io import StringIO
import pyomo.core.expr as EXPR
from pyomo.repn import generate_standard_repn
import logging
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy import __version__
from pyomo.common.dependencies import attempt_import
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.common.collections import ComponentMap, Bunch, ComponentSet
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.mindtpy.cut_generation import add_no_good_cuts
from operator import itemgetter
from pyomo.common.errors import DeveloperError
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.opt import (
from pyomo.core import (
from pyomo.contrib.gdpopt.util import (
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.contrib.mindtpy.util import (
def solve_main(self):
    """This function solves the MIP main problem.

        Returns
        -------
        self.mip : Pyomo model
            The MIP stored in self.
        main_mip_results : SolverResults
            Results from solving the main MIP.
        """
    config = self.config
    self.mip_iter += 1
    self.setup_main()
    mip_args = self.set_up_mip_solver()
    update_solver_timelimit(self.mip_opt, config.mip_solver, self.timing, config)
    try:
        main_mip_results = self.mip_opt.solve(self.mip, tee=config.mip_solver_tee, load_solutions=self.load_solutions, **mip_args)
        if len(main_mip_results.solution) > 0:
            self.mip.solutions.load_from(main_mip_results)
    except (ValueError, AttributeError, RuntimeError) as e:
        config.logger.error(e, exc_info=True)
        if config.single_tree:
            config.logger.warning('Single tree terminate.')
            if get_main_elapsed_time(self.timing) >= config.time_limit:
                config.logger.warning('due to the timelimit.')
                self.results.solver.termination_condition = tc.maxTimeLimit
            if config.strategy == 'GOA' or config.add_no_good_cuts:
                config.logger.warning("Error: Cannot load a SolverResults object with bad status: error. MIP solver failed. This usually happens in the single-tree GOA algorithm. No-good cuts are added and GOA algorithm doesn't converge within the time limit. No integer solution is found, so the CPLEX solver will report an error status. ")
        if 'main_mip_results' in locals():
            return (self.mip, main_mip_results)
        else:
            return (None, None)
    if config.solution_pool:
        main_mip_results._solver_model = self.mip_opt._solver_model
        main_mip_results._pyomo_var_to_solver_var_map = self.mip_opt._pyomo_var_to_solver_var_map
    if main_mip_results.solver.termination_condition is tc.optimal:
        if config.single_tree and (not config.add_no_good_cuts):
            self.update_suboptimal_dual_bound(main_mip_results)
    elif main_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
        main_mip_results, _ = distinguish_mip_infeasible_or_unbounded(self.mip, config)
    return (self.mip, main_mip_results)