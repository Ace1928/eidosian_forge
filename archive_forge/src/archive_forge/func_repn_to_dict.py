import pickle
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types, as_numeric, value
from pyomo.core.expr.visitor import replace_expressions
from pyomo.repn import generate_standard_repn
from pyomo.environ import (
import pyomo.kernel
def repn_to_dict(repn):
    result = {}
    for i in range(len(repn.linear_vars)):
        if id(repn.linear_vars[i]) in result:
            result[id(repn.linear_vars[i])] += value(repn.linear_coefs[i])
        else:
            result[id(repn.linear_vars[i])] = value(repn.linear_coefs[i])
    for i in range(len(repn.quadratic_vars)):
        v1_, v2_ = repn.quadratic_vars[i]
        if id(v1_) <= id(v2_):
            result[id(v1_), id(v2_)] = value(repn.quadratic_coefs[i])
        else:
            result[id(v2_), id(v1_)] = value(repn.quadratic_coefs[i])
    if not (repn.constant is None or (type(repn.constant) in native_numeric_types and repn.constant == 0)):
        result[None] = value(repn.constant)
    return result