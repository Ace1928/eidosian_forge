from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.norm import norm
from cvxpy.expressions.expression import Expression
def tv(value, *args):
    """Total variation of a vector, matrix, or list of matrices.

    Uses L1 norm of discrete gradients for vectors and
    L2 norm of discrete gradients for matrices.

    Parameters
    ----------
    value : Expression or numeric constant
        The value to take the total variation of.
    args : Matrix constants/expressions
        Additional matrices extending the third dimension of value.

    Returns
    -------
    Expression
        An Expression representing the total variation.
    """
    value = Expression.cast_to_const(value)
    if value.ndim == 0:
        raise ValueError('tv cannot take a scalar argument.')
    elif value.ndim == 1:
        return norm(value[1:] - value[0:value.shape[0] - 1], 1)
    else:
        rows, cols = value.shape
        args = map(Expression.cast_to_const, args)
        values = [value] + list(args)
        diffs = []
        for mat in values:
            diffs += [mat[0:rows - 1, 1:cols] - mat[0:rows - 1, 0:cols - 1], mat[1:rows, 0:cols - 1] - mat[0:rows - 1, 0:cols - 1]]
        length = diffs[0].shape[0] * diffs[1].shape[1]
        stacked = vstack([reshape(diff, (1, length)) for diff in diffs])
        return sum(norm(stacked, p=2, axis=0))