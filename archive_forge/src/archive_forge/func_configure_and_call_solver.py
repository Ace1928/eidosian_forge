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
def configure_and_call_solver(model, solver, args, problem_type, timing, time_limit):
    opt = SolverFactory(solver)
    if not opt.available():
        raise RuntimeError('%s solver %s is not available.' % (problem_type, solver))
    with SuppressInfeasibleWarning():
        solver_args = dict(args)
        if time_limit is not None:
            elapsed = get_main_elapsed_time(timing)
            remaining = max(time_limit - elapsed, 1)
            if solver == 'gams':
                solver_args['add_options'] = solver_args.get('add_options', [])
                solver_args['add_options'].append('option reslim=%s;' % remaining)
            elif solver == 'multisolve':
                solver_args['time_limit'] = min(solver_args.get('time_limit', float('inf')), remaining)
        try:
            results = opt.solve(model, **solver_args)
        except ValueError as err:
            if 'Cannot load a SolverResults object with bad status: error' in str(err):
                results = SolverResults()
                results.solver.termination_condition = tc.error
                results.solver.message = str(err)
            else:
                raise
    return results