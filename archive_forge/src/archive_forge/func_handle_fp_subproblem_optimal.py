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
def handle_fp_subproblem_optimal(self, fp_nlp):
    """Copies the solution to the working model, updates bound, adds OA cuts / no-good cuts /
        increasing objective cut, calculates the duals and stores incumbent solution if it has been improved.

        Parameters
        ----------
        fp_nlp : Pyomo model
            The feasibility pump NLP subproblem.
        """
    copy_var_list_values(fp_nlp.MindtPy_utils.variable_list, self.working_model.MindtPy_utils.variable_list, self.config, ignore_integrality=True)
    add_orthogonality_cuts(self.working_model, self.mip, self.config)
    if fp_converged(self.working_model, self.mip, proj_zero_tolerance=self.config.fp_projzerotol, discrete_only=self.config.fp_discrete_only):
        copy_var_list_values(self.mip.MindtPy_utils.variable_list, self.fixed_nlp.MindtPy_utils.variable_list, self.config, skip_fixed=False)
        fixed_nlp, fixed_nlp_results = self.solve_subproblem()
        if fixed_nlp_results.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
            self.handle_subproblem_optimal(fixed_nlp)
            if self.primal_bound_improved:
                self.mip.MindtPy_utils.cuts.del_component('improving_objective_cut')
                if self.objective_sense == minimize:
                    self.mip.MindtPy_utils.cuts.improving_objective_cut = Constraint(expr=sum(self.mip.MindtPy_utils.objective_value[:]) <= self.primal_bound - self.config.fp_cutoffdecr * max(1, abs(self.primal_bound)))
                else:
                    self.mip.MindtPy_utils.cuts.improving_objective_cut = Constraint(expr=sum(self.mip.MindtPy_utils.objective_value[:]) >= self.primal_bound + self.config.fp_cutoffdecr * max(1, abs(self.primal_bound)))
        else:
            self.config.logger.error('Feasibility pump Fixed-NLP is infeasible, something might be wrong. There might be a problem with the precisions - the feasibility pump seems to have converged')