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
def solve_fp_subproblem(self):
    """Solves the feasibility pump NLP subproblem.

        This function sets up the 'fp_nlp' by relax integer variables.
        precomputes dual values, deactivates trivial constraints, and then solves NLP model.

        Returns
        -------
        fp_nlp : Pyomo model
            Fixed-NLP from the model.
        results : SolverResults
            Results from solving the fixed-NLP subproblem.
        """
    fp_nlp = self.working_model.clone()
    MindtPy = fp_nlp.MindtPy_utils
    config = self.config
    fp_nlp.MindtPy_utils.objective_list[-1].deactivate()
    if self.objective_sense == minimize:
        fp_nlp.improving_objective_cut = Constraint(expr=sum(fp_nlp.MindtPy_utils.objective_value[:]) <= self.primal_bound)
    else:
        fp_nlp.improving_objective_cut = Constraint(expr=sum(fp_nlp.MindtPy_utils.objective_value[:]) >= self.primal_bound)
    if config.fp_norm_constraint:
        generate_norm_constraint(fp_nlp, self.mip, config)
    MindtPy.fp_nlp_obj = generate_norm2sq_objective_function(fp_nlp, self.mip, discrete_only=config.fp_discrete_only)
    MindtPy.cuts.deactivate()
    TransformationFactory('core.relax_integer_vars').apply_to(fp_nlp)
    try:
        TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(fp_nlp, tmp=True, ignore_infeasible=False, tolerance=config.constraint_tolerance)
    except InfeasibleConstraintException as e:
        config.logger.error(e, exc_info=True)
        config.logger.error('Infeasibility detected in deactivate_trivial_constraints.')
        results = SolverResults()
        results.solver.termination_condition = tc.infeasible
        return (fp_nlp, results)
    nlp_args = dict(config.nlp_solver_args)
    update_solver_timelimit(self.nlp_opt, config.nlp_solver, self.timing, config)
    with SuppressInfeasibleWarning():
        with time_code(self.timing, 'fp subproblem'):
            results = self.nlp_opt.solve(fp_nlp, tee=config.nlp_solver_tee, load_solutions=self.load_solutions, **nlp_args)
            if len(results.solution) > 0:
                fp_nlp.solutions.load_from(results)
    return (fp_nlp, results)