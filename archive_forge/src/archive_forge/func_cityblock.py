import cupy
def cityblock(u, v):
    """Compute the City Block (Manhattan) distance between two 1-D arrays.

    The City Block distance is defined as

    .. math::
        d(u, v) = \\sum_{i} |u_i - v_i|

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        cityblock (double): The City Block distance between
        vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, 'cityblock')
    return output_arr[0]