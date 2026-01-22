import collections
import enum
import functools
import itertools
import logging
import operator
import sys
from pyomo.common.collections import Sequence, ComponentMap, ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.expr.numvalue import is_fixed, value
import pyomo.core.expr as EXPR
import pyomo.core.kernel as kernel
def complex_number_error(value, visitor, expr, node=''):
    msg = f'Pyomo {visitor.__class__.__name__} does not support complex numbers'
    cause = ' '.join(filter(None, ('Complex number returned from expression', node)))
    logger.warning(f'{cause}\n\tmessage: {msg}\n\texpression: {expr}')
    if HALT_ON_EVALUATION_ERROR:
        raise InvalidValueError(f'Pyomo {visitor.__class__.__name__} does not support complex numbers')
    return InvalidNumber(value, cause)