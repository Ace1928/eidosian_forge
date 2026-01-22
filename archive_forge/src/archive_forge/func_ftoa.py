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
def ftoa(val, parenthesize_negative_values=False):
    if val is None:
        return val
    if type(val) in native_numeric_types:
        _val = val
    elif is_fixed(val):
        _val = value(val)
    else:
        raise ValueError('Converting non-fixed bound or value to string: %s' % (val,))
    a = _ftoa_precision_str % _val
    i = len(a) - 1
    if i:
        try:
            while float(a[:i]) == _val:
                i -= 1
        except:
            pass
    i += 1
    if i == len(a) and float(a) != _val:
        logger.warning('Converting %s to string resulted in loss of precision' % val)
    if parenthesize_negative_values and a[0] == '-':
        return '(' + a[:i] + ')'
    else:
        return a[:i]