from __future__ import annotations
import numbers
from typing import List, Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.utilities.shape import size_from_shape
def deep_flatten(x):
    if isinstance(x, Expression):
        if len(x.shape) == 1:
            return x
        else:
            return x.flatten()
    elif isinstance(x, np.ndarray) or isinstance(x, (int, float)):
        x = Expression.cast_to_const(x)
        return x.flatten()
    if isinstance(x, list):
        y = []
        for x0 in x:
            x1 = deep_flatten(x0)
            y.append(x1)
        y = hstack(y)
        return y
    msg = 'The input to deep_flatten must be an Expression, a NumPy array, an int' + ' or float, or a nested list thereof. Received input of type %s' % type(x)
    raise ValueError(msg)