import cupy
def russellrao(u, v):
    """Compute the Russell-Rao distance between two 1-D arrays.

    The Russell-Rao distance is defined as the proportion of elements
    in both `u` and `v` that are in the exact same position:

    .. math::
        d(u, v) = \\frac{1}{n} \\sum_{k=0}^n u_i = v_i

    where :math:`x = y` is one if :math:`x` is different from :math:`y`
    and zero otherwise.

    Args:
        u (array_like): Input array of size (N,)
        v (array_like): Input array of size (N,)

    Returns:
        hamming (double): The Hamming distance between vectors `u` and `v`.
    """
    if not pylibraft_available:
        raise RuntimeError('pylibraft is not installed')
    u = cupy.asarray(u)
    v = cupy.asarray(v)
    output_arr = cupy.zeros((1,), dtype=u.dtype)
    pairwise_distance(u, v, output_arr, 'russellrao')
    return output_arr[0]