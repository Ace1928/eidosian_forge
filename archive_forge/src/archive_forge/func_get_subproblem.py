from pyomo.core import (
from pyomo.core.base import TransformationFactory, Suffix, ConstraintList, Integers
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.util import (
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.util.vars_from_expressions import get_vars_from_components
def get_subproblem(original_model, util_block):
    """Clone the original, and reclassify all the Disjuncts to Blocks.
    We'll also call logical_to_disjunctive and bigm the disjunctive parts in
    case any of the indicator_vars are used in logical constraints and to make
    sure that the rest of the model is algebraic (assuming it was a proper
    GDP to begin with).
    """
    subproblem = original_model.clone()
    subproblem.name = subproblem.name + ': subproblem'
    if not hasattr(subproblem, 'dual'):
        subproblem.dual = Suffix(direction=Suffix.IMPORT)
    elif not isinstance(subproblem.dual, Suffix):
        raise ValueError("The model contains a component called 'dual' that is not a Suffix. It is of type %s. Please rename this component, as GDPopt needs dual information to create cuts." % type(subproblem.dual))
    subproblem.dual.activate()
    for disjunction in subproblem.component_data_objects(Disjunction, descend_into=(Block, Disjunct), descent_order=TraversalStrategy.PostfixDFS):
        for disjunct in disjunction.disjuncts:
            if disjunct.indicator_var.fixed:
                if not disjunct.indicator_var.value:
                    disjunct.deactivate()
            disjunct.parent_block().reclassify_component_type(disjunct, Block)
        disjunction.deactivate()
    TransformationFactory('contrib.logical_to_disjunctive').apply_to(subproblem)
    TransformationFactory('gdp.bigm').apply_to(subproblem)
    subproblem_util_block = subproblem.component(util_block.local_name)
    save_initial_values(subproblem_util_block)
    add_transformed_boolean_variable_list(subproblem_util_block)
    subproblem_obj = next(subproblem.component_data_objects(Objective, active=True, descend_into=True))
    subproblem_util_block.obj = Expression(expr=subproblem_obj.expr)
    return (subproblem, subproblem_util_block)