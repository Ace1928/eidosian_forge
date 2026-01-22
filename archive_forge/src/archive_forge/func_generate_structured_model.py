from logging import getLogger
from pyomo.common.dependencies import attempt_import
from pyomo.core import (
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.expr.visitor import replace_expressions, identify_variables
from pyomo.contrib.community_detection.community_graph import generate_model_graph
from pyomo.common.dependencies import networkx as nx
from pyomo.common.dependencies.matplotlib import pyplot as plt
from itertools import combinations
import copy
def generate_structured_model(self):
    """
        Using the community map and the original model used to create this community map, we will create
        structured_model, which will be based on the original model but will place variables, constraints, and
        objectives into or outside of various blocks (communities) based on the community map.

        Returns
        -------
        structured_model: Block
            a Pyomo model that reflects the nature of the community map
        """
    structured_model = ConcreteModel()
    structured_model.b = Block([0, len(self.community_map) - 1, 1])
    blocked_variable_map = ComponentMap()
    for community_key, community in self.community_map.items():
        _, variables_in_community = community
        for stored_variable in variables_in_community:
            new_variable = Var(domain=stored_variable.domain, bounds=stored_variable.bounds)
            structured_model.b[community_key].add_component(str(stored_variable), new_variable)
            variable_in_new_model = structured_model.find_component(new_variable)
            blocked_variable_map[stored_variable] = blocked_variable_map.get(stored_variable, []) + [variable_in_new_model]
    replace_variables_in_expression_map = dict()
    for community_key, community in self.community_map.items():
        constraints_in_community, _ = community
        for stored_constraint in constraints_in_community:
            for variable_in_stored_constraint in identify_variables(stored_constraint.expr):
                variable_in_current_block = False
                for blocked_variable in blocked_variable_map[variable_in_stored_constraint]:
                    if 'b[%d]' % community_key in str(blocked_variable):
                        replace_variables_in_expression_map[id(variable_in_stored_constraint)] = blocked_variable
                        variable_in_current_block = True
                if not variable_in_current_block:
                    new_variable = Var(domain=variable_in_stored_constraint.domain, bounds=variable_in_stored_constraint.bounds)
                    structured_model.add_component(str(variable_in_stored_constraint), new_variable)
                    variable_in_new_model = structured_model.find_component(new_variable)
                    blocked_variable_map[variable_in_stored_constraint] = blocked_variable_map.get(variable_in_stored_constraint, []) + [variable_in_new_model]
                    replace_variables_in_expression_map[id(variable_in_stored_constraint)] = variable_in_new_model
            if self.with_objective and isinstance(stored_constraint, (_GeneralObjectiveData, Objective)):
                new_objective = Objective(expr=replace_expressions(stored_constraint.expr, replace_variables_in_expression_map))
                structured_model.b[community_key].add_component(str(stored_constraint), new_objective)
            else:
                new_constraint = Constraint(expr=replace_expressions(stored_constraint.expr, replace_variables_in_expression_map))
                structured_model.b[community_key].add_component(str(stored_constraint), new_constraint)
    if not self.with_objective:
        for objective_function in self.model.component_data_objects(ctype=Objective, active=self.use_only_active_components, descend_into=True):
            for variable_in_objective in identify_variables(objective_function):
                if structured_model.find_component(str(variable_in_objective)) is None:
                    new_variable = Var(domain=variable_in_objective.domain, bounds=variable_in_objective.bounds)
                    structured_model.add_component(str(variable_in_objective), new_variable)
                    variable_in_new_model = structured_model.find_component(new_variable)
                    blocked_variable_map[variable_in_objective] = blocked_variable_map.get(variable_in_objective, []) + [variable_in_new_model]
                    replace_variables_in_expression_map[id(variable_in_objective)] = variable_in_new_model
                else:
                    for version_of_variable in blocked_variable_map[variable_in_objective]:
                        if 'b[' not in str(version_of_variable):
                            replace_variables_in_expression_map[id(variable_in_objective)] = version_of_variable
            new_objective = Objective(expr=replace_expressions(objective_function.expr, replace_variables_in_expression_map))
            structured_model.add_component(str(objective_function), new_objective)
    structured_model.equality_constraint_list = ConstraintList(doc='Equality Constraints for the different forms of a given variable')
    for variable, duplicate_variables in blocked_variable_map.items():
        equalities_to_make = combinations(duplicate_variables, 2)
        for variable_1, variable_2 in equalities_to_make:
            structured_model.equality_constraint_list.add(expr=variable_1 == variable_2)
    return structured_model