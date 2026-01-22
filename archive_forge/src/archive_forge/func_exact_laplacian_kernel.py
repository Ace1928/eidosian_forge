from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def exact_laplacian_kernel(x, y, stddev):
    """Computes exact Laplacian kernel value(s) for tensors x and y using stddev.

  The Laplacian kernel for vectors u, v is defined as follows:
       K(u, v) = exp(-||u-v|| / stddev)
  where the norm is the l1-norm. x, y can be either vectors or matrices. If they
  are vectors, they must have the same dimension. If they are matrices, they
  must have the same number of columns. In the latter case, the method returns
  (as a matrix) K(u, v) values for all pairs (u, v) where u is a row from x and
  v is a row from y.

  Args:
    x: a tensor of rank 1 or 2. It's shape should be either [dim] or [m, dim].
    y: a tensor of rank 1 or 2. It's shape should be either [dim] or [n, dim].
    stddev: The width of the Gaussian kernel.

  Returns:
    A single value (scalar) with shape (1, 1)  if x, y are vectors or a matrix
    of shape (m, n) with entries K(u, v) (where K is the Laplacian kernel) for
    all (u,v) pairs where u, v are rows from x and y respectively.

  Raises:
    ValueError: if the shapes of x, y are not compatible.
  """
    x_aligned, y_aligned = _align_matrices(x, y)
    diff_l1_norm = math_ops.reduce_sum(math_ops.abs(math_ops.subtract(x_aligned, y_aligned)), 2)
    return math_ops.exp(-diff_l1_norm / stddev)