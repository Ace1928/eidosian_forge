import numpy as np
import ray
import ray.experimental.array.remote as ra
from . import core
@ray.remote(num_returns=3)
def modified_lu(q):
    """Perform a modified LU decomposition of a matrix.

    This takes a matrix q with orthonormal columns, returns l, u, s such that
    q - s = l * u.

    Args:
        q: A two dimensional orthonormal matrix q.

    Returns:
        A tuple of a lower triangular matrix l, an upper triangular matrix u,
            and a a vector representing a diagonal matrix s such that
            q - s = l * u.
    """
    q = q.assemble()
    m, b = (q.shape[0], q.shape[1])
    S = np.zeros(b)
    q_work = np.copy(q)
    for i in range(b):
        S[i] = -1 * np.sign(q_work[i, i])
        q_work[i, i] -= S[i]
        q_work[i + 1:m, i] /= q_work[i, i]
        q_work[i + 1:m, i + 1:b] -= np.outer(q_work[i + 1:m, i], q_work[i, i + 1:b])
    L = np.tril(q_work)
    for i in range(b):
        L[i, i] = 1
    U = np.triu(q_work)[:b, :]
    return (ray.get(core.numpy_to_dist.remote(ray.put(L))), U, S)