def setup_ldl_factor(c, G, h, dims, A, b):
    """
    The meanings of arguments in this function are identical to those of the
    function cvxopt.solvers.conelp. Refer to CVXOPT documentation

        https://cvxopt.org/userguide/coneprog.html#linear-cone-programs

    for more information.

    Note: CVXOPT allows G and A to be passed as dense matrix objects. However,
    this function will only ever be called with spmatrix objects. If creating
    a custom kktsolver of your own, you need to conform to this sparse matrix
    assumption.
    """
    factor = kkt_ldl(G, dims, A)
    return factor