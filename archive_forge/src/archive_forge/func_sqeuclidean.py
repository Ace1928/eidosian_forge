import cupy
def sqeuclidean(u, v):
    """Compute the squared Euclidean distance between two 1-D arrays.

    The squared Euclidean distance is defined as

    .. math::
        d(u, v) = \\sum_{i} (u_i - v_i)^2

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        sqeuclidean (double): The squared Euclidean distance between
        vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed.')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, 'sqeuclidean')
    return output_arr[0]