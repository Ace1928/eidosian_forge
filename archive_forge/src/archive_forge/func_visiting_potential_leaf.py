import itertools
import logging
import math
from io import StringIO
from contextlib import nullcontext
from pyomo.common.collections import OrderedSet
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.visitor import _ToStringVisitor
import pyomo.core.expr as EXPR
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
import pyomo.core.base.suffix
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.repn.util import valid_expr_ctypes_minlp, valid_active_ctypes_minlp, ftoa
def visiting_potential_leaf(self, node):
    """
        Visiting a potential leaf.

        Return True if the node is not expanded.
        """
    if node.__class__ in native_types:
        return (True, ftoa(node, True))
    if node.is_expression_type():
        if not node.is_potentially_variable():
            return (True, ftoa(node(), True))
        if node.__class__ is EXPR.MonomialTermExpression:
            return (True, self._monomial_to_string(node))
        if node.__class__ is EXPR.LinearExpression:
            return (True, self._linear_to_string(node))
        return (False, None)
    if node.is_component_type():
        if node.ctype not in valid_expr_ctypes_minlp:
            raise RuntimeError("Unallowable component '%s' of type %s found in an active constraint or objective.\nThe GAMS writer cannot export expressions with this component type." % (node.name, node.ctype.__name__))
    if node.is_fixed():
        return (True, ftoa(node(), True))
    else:
        assert node.is_variable_type()
        self.variables.add(id(node))
        return (True, self.smap.getSymbol(node))