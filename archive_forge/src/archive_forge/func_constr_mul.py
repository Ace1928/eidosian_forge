import numpy as np
from cvxpy.lin_ops.tree_mat import mul, sum_dicts, tmul
def constr_mul(constraints, var_dict, vec_size, is_abs):
    """Multiplies a vector by the matrix implied by the constraints.

    Parameters
    ----------
    constraints : list
        A list of linear constraints.
    var_dict : dict
        A dictionary mapping variable id to value.
    vec_size : int
        The length of the product vector.
    is_abs : bool
        Multiply by the absolute value of the matrix?
    """
    product = np.zeros(vec_size)
    offset = 0
    for constr in constraints:
        result = mul(constr.expr, var_dict, is_abs)
        rows, cols = constr.size
        for col in range(cols):
            if np.isscalar(result):
                product[offset:offset + rows] = result
            else:
                product[offset:offset + rows] = np.squeeze(result[:, col])
            offset += rows
    return product