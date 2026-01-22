from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import InfeasibleConstraintException, DeveloperError
from pyomo.contrib import appsi
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.contrib.gdpopt.util import (
from pyomo.core import Constraint, TransformationFactory, Objective, Block
import pyomo.core.expr as EXPR
from pyomo.opt import SolverFactory, SolverResults
from pyomo.opt import TerminationCondition as tc
def solve_NLP(nlp_model, config, timing):
    """Solve the NLP subproblem."""
    config.logger.debug('Solving nonlinear subproblem for fixed binaries and logical realizations.')
    results = configure_and_call_solver(nlp_model, config.nlp_solver, config.nlp_solver_args, 'NLP', timing, config.time_limit)
    return process_nonlinear_problem_results(results, nlp_model, 'NLP', config)