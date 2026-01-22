from pyomo.core import (
from pyomo.core.base import TransformationFactory, Suffix, ConstraintList, Integers
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.util import (
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.util.vars_from_expressions import get_vars_from_components
def initialize_discrete_problem(util_block, subprob_util_block, config, solver):
    """
    Calls the specified transformation (by default bigm) on the original
    model and removes nonlinear constraints to create a MILP discrete problem.
    """
    config.logger.info('---Starting GDPopt initialization---')
    discrete = util_block.parent_block().clone()
    discrete.name = discrete.name + ': discrete problem'
    discrete_problem_util_block = discrete.component(util_block.local_name)
    discrete_problem_util_block.no_good_cuts = ConstraintList()
    discrete_problem_util_block.no_good_disjunctions = Disjunction(Integers)
    for c in discrete.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)):
        if c.body.polynomial_degree() not in (1, 0):
            c.deactivate()
    TransformationFactory(config.discrete_problem_transformation).apply_to(discrete)
    add_transformed_boolean_variable_list(discrete_problem_util_block)
    add_algebraic_variable_list(discrete_problem_util_block, name='all_mip_variables')
    init_algorithm = valid_init_strategies.get(config.init_algorithm)
    init_algorithm(util_block, discrete_problem_util_block, subprob_util_block, config, solver)
    return discrete_problem_util_block