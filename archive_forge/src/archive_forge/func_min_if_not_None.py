from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.base import Block, Constraint, VarList, Objective, TransformationFactory
from pyomo.core.expr import ExpressionReplacementVisitor
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
import logging
def min_if_not_None(iterable):
    """Returns the minimum among non-None elements.

    Returns None when all elements are None.

    """
    non_nones = [a for a in iterable if a is not None]
    return min(non_nones or [None])