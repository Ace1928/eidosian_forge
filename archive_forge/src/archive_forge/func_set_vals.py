from __future__ import annotations
from typing import Tuple
import numpy as np
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.constants.parameter import is_param_free
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.utilities import scopes
def set_vals(vals, s_val):
    for var, val in zip(f.variables(), vals):
        var.value = val / s_val