from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt, BoundsManager
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.core.expr import identify_variables
from pyomo.core import Constraint, Objective, TransformationFactory, minimize, value
from pyomo.opt import SolverFactory
from pyomo.gdp.disjunct import Disjunct
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.opt import TerminationCondition as tc
def solve_bounding_problem(model, solver):
    results = SolverFactory(solver).solve(model)
    if results.solver.termination_condition is tc.optimal:
        return value(model._var_bounding_obj.expr)
    elif results.solver.termination_condition is tc.infeasible:
        return None
    elif results.solver.termination_condition is tc.unbounded:
        return -inf
    else:
        raise NotImplementedError('Unhandled termination condition: %s' % results.solver.termination_condition)