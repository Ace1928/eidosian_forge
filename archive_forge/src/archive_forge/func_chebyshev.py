import cupy
def chebyshev(u, v):
    """Compute the Chebyshev distance between two 1-D arrays.

    The Chebyshev distance is defined as

    .. math::
        d(u, v) = \\max_{i} |u_i - v_i|

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        chebyshev (double): The Chebyshev distance between vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, 'chebyshev')
    return output_arr[0]