from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def exact_gaussian_kernel(x, y, stddev):
    """Computes exact Gaussian kernel value(s) for tensors x and y and stddev.

  The Gaussian kernel for vectors u, v is defined as follows:
       K(u, v) = exp(-||u-v||^2 / (2* stddev^2))
  where the norm is the l2-norm. x, y can be either vectors or matrices. If they
  are vectors, they must have the same dimension. If they are matrices, they
  must have the same number of columns. In the latter case, the method returns
  (as a matrix) K(u, v) values for all pairs (u, v) where u is a row from x and
  v is a row from y.

  Args:
    x: a tensor of rank 1 or 2. It's shape should be either [dim] or [m, dim].
    y: a tensor of rank 1 or 2. It's shape should be either [dim] or [n, dim].
    stddev: The width of the Gaussian kernel.

  Returns:
    A single value (scalar) with shape (1, 1) (if x, y are vectors) or a matrix
      of shape (m, n) with entries K(u, v) (where K is the Gaussian kernel) for
      all (u,v) pairs where u, v are rows from x and y respectively.

  Raises:
    ValueError: if the shapes of x, y are not compatible.
  """
    x_aligned, y_aligned = _align_matrices(x, y)
    diff_squared_l2_norm = math_ops.reduce_sum(math_ops.squared_difference(x_aligned, y_aligned), 2)
    return math_ops.exp(-diff_squared_l2_norm / (2 * stddev * stddev))