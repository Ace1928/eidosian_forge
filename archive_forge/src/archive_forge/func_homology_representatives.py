from . import matrix
def homology_representatives(d1, d2, N):
    """
    Given two matrices d1 and d2 such that d1 * d2 = 0, computes the
    homology H_1 = ker(d1) / im(d2) when using Z/N coefficients.

    The result is a list of vectors c_1 for each homology class [c_1].
    """
    assert N > 1
    homology_basis = homology_basis_representatives_with_orders(d1, d2, N)
    return _enumerate_from_basis(homology_basis, matrix.num_cols(d1), N)