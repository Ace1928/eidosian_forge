from typing import Tuple
import scipy.sparse as sp
import cvxpy.lin_ops.lin_utils as lu
def format_elemwise(vars_):
    """Formats all the elementwise cones for the solver.

    Parameters
    ----------
    vars_ : list
        A list of the LinOp expressions in the elementwise cones.

    Returns
    -------
    list
        A list of LinLeqConstr that represent all the elementwise cones.
    """
    spacing = len(vars_)
    mat_shape = (spacing * vars_[0].shape[0], vars_[0].shape[0])
    terms = []
    for i, var in enumerate(vars_):
        mat = get_spacing_matrix(mat_shape, spacing, i)
        terms.append(lu.mul_expr(mat, var))
    return [lu.create_geq(lu.sum_expr(terms))]