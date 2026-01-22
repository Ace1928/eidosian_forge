from cvxpy.atoms.lambda_sum_largest import lambda_sum_largest
from cvxpy.expressions.expression import Expression
def lambda_sum_smallest(X, k):
    """Sum of the largest k eigenvalues.
    """
    X = Expression.cast_to_const(X)
    return -lambda_sum_largest(-X, k)