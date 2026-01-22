import ctypes
import logging
import os
from collections import deque, defaultdict, namedtuple
from contextlib import nullcontext
from itertools import filterfalse, product
from math import log10 as _log10
from operator import itemgetter, attrgetter, setitem
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException, MouseTrap
from pyomo.common.gc_manager import PauseGC
from pyomo.common.numeric_types import (
from pyomo.common.timing import TicTocTimer
from pyomo.core.expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import (
from pyomo.core.base.suffix import SuffixFinder
from pyomo.core.base.var import _VarData
import pyomo.core.kernel as kernel
from pyomo.core.pyomoobject import PyomoObject
from pyomo.opt import WriterFactory
from pyomo.repn.util import (
from pyomo.repn.plugins.ampl.ampl_ import set_pyomo_amplfunc_env
from pyomo.core.base import Set, RangeSet
from pyomo.network import Port
def handle_product_node(visitor, node, arg1, arg2):
    if arg2[0] is _CONSTANT:
        arg2, arg1 = (arg1, arg2)
    if arg1[0] is _CONSTANT:
        mult = arg1[1]
        if not mult:
            if arg2[0] is _CONSTANT:
                _prod = mult * arg2[1]
                if _prod:
                    deprecation_warning(f'Encountered {mult}*{str(arg2[1])} in expression tree.  Mapping the NaN result to 0 for compatibility with the nl_v1 writer.  In the future, this NaN will be preserved/emitted to comply with IEEE-754.', version='6.4.3')
                    _prod = 0
                return (_CONSTANT, _prod)
            return arg1
        if mult == 1:
            return arg2
        elif arg2[0] is _MONOMIAL:
            if mult != mult:
                return arg1
            return (_MONOMIAL, arg2[1], mult * arg2[2])
        elif arg2[0] is _GENERAL:
            if mult != mult:
                return arg1
            arg2[1].mult *= mult
            return arg2
        elif arg2[0] is _CONSTANT:
            if not arg2[1]:
                _prod = mult * arg2[1]
                if _prod:
                    deprecation_warning(f'Encountered {str(mult)}*{arg2[1]} in expression tree.  Mapping the NaN result to 0 for compatibility with the nl_v1 writer.  In the future, this NaN will be preserved/emitted to comply with IEEE-754.', version='6.4.3')
                    _prod = 0
                return (_CONSTANT, _prod)
            return (_CONSTANT, mult * arg2[1])
    nonlin = node_result_to_amplrepn(arg1).compile_repn(visitor, visitor.template.product)
    nonlin = node_result_to_amplrepn(arg2).compile_repn(visitor, *nonlin)
    return (_GENERAL, AMPLRepn(0, None, nonlin))