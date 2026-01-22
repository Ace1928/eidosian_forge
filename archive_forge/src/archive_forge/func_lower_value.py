import numpy as np
import scipy.sparse as sp
from cvxpy.atoms import diag, reshape
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable, upper_tri_to_full
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution
def lower_value(variable, value):
    if attributes_present([variable], SYMMETRIC_ATTRIBUTES):
        return value[np.triu_indices(variable.shape[0])]
    elif variable.attributes['diag']:
        return np.diag(value)
    else:
        return value