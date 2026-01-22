from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.base import Block, Constraint, VarList, Objective, TransformationFactory
from pyomo.core.expr import ExpressionReplacementVisitor
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
import logging
Update the values of the variables that were replaced by aggregates.

        TODO: reduced costs

        