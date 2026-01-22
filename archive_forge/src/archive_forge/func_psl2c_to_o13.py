from ..matrix import matrix
def psl2c_to_o13(A):
    """
    Converts matrix in PSL(2,C) to O13.

    Python implementation of Moebius_to_O31 in matrix_conversion.c.
    """
    return matrix([_o13_matrix_column(A, m) for m in _basis_vectors_sl2c(A.base_ring())]).transpose()