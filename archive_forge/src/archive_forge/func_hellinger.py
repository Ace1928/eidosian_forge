import cupy
def hellinger(u, v):
    """Compute the Hellinger distance between two 1-D arrays.

    The Hellinger distance is defined as

    .. math::
        d(u, v) = \\frac{1}{\\sqrt{2}} \\sqrt{
            \\sum_{i} (\\sqrt{u_i} - \\sqrt{v_i})^2}

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        hellinger (double): The Hellinger distance between
        vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, 'hellinger')
    return output_arr[0]