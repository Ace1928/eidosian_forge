from pyomo.core.base.block import Block
from pyomo.core.base.reference import Reference
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.expression import Expression
from pyomo.core.base.external import ExternalFunction
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types

        Parameters
        ----------
        n_scenario: Integer
            The number of different values we expect for each input variable
        input_values: ComponentMap
            Maps each input variable to a list of values of length n_scenario
        output_values: ComponentMap
            Maps each output variable to a list of values of length n_scenario
        to_fix: List
            to_fix argument for base class
        to_deactivate: List
            to_deactivate argument for base class
        to_reset: List
            to_reset argument for base class. This list is extended with
            input variables.

        