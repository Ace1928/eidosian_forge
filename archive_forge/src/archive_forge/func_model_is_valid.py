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
def model_is_valid(self):
    """Determines whether the model is solvable by MindtPy.

        Returns
        -------
        bool
            True if model is solvable in MindtPy, False otherwise.
        """
    m = self.working_model
    MindtPy = m.MindtPy_utils
    config = self.config
    prob = self.results.problem
    if len(MindtPy.discrete_variable_list) == 0:
        config.logger.info('Problem has no discrete decisions.')
        obj = next(m.component_data_objects(ctype=Objective, active=True))
        if any((c.body.polynomial_degree() not in self.mip_constraint_polynomial_degree for c in MindtPy.constraint_list)) or obj.expr.polynomial_degree() not in self.mip_objective_polynomial_degree:
            config.logger.info('Your model is a NLP (nonlinear program). Using NLP solver %s to solve.' % config.nlp_solver)
            update_solver_timelimit(self.nlp_opt, config.nlp_solver, self.timing, config)
            self.nlp_opt.solve(self.original_model, tee=config.nlp_solver_tee, **config.nlp_solver_args)
            return False
        else:
            config.logger.info('Your model is an LP (linear program). Using LP solver %s to solve.' % config.mip_solver)
            if isinstance(self.mip_opt, PersistentSolver):
                self.mip_opt.set_instance(self.original_model)
            update_solver_timelimit(self.mip_opt, config.mip_solver, self.timing, config)
            results = self.mip_opt.solve(self.original_model, tee=config.mip_solver_tee, load_solutions=self.load_solutions, **config.mip_solver_args)
            if len(results.solution) > 0:
                self.original_model.solutions.load_from(results)
            return False
    if config.calculate_dual_at_solution:
        if not hasattr(m, 'dual'):
            m.dual = Suffix(direction=Suffix.IMPORT)
        elif not isinstance(m.dual, Suffix):
            raise ValueError('dual is not defined as a Suffix in the original model.')
    return True