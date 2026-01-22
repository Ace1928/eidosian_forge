from ._base import issparse
from ._csr import csr_array
from ._sparsetools import csr_count_blocks
def estimate_blocksize(A, efficiency=0.7):
    """Attempt to determine the blocksize of a sparse matrix

    Returns a blocksize=(r,c) such that
        - A.nnz / A.tobsr( (r,c) ).nnz > efficiency
    """
    if not (issparse(A) and A.format in ('csc', 'csr')):
        A = csr_array(A)
    if A.nnz == 0:
        return (1, 1)
    if not 0 < efficiency < 1.0:
        raise ValueError('efficiency must satisfy 0.0 < efficiency < 1.0')
    high_efficiency = (1.0 + efficiency) / 2.0
    nnz = float(A.nnz)
    M, N = A.shape
    if M % 2 == 0 and N % 2 == 0:
        e22 = nnz / (4 * count_blocks(A, (2, 2)))
    else:
        e22 = 0.0
    if M % 3 == 0 and N % 3 == 0:
        e33 = nnz / (9 * count_blocks(A, (3, 3)))
    else:
        e33 = 0.0
    if e22 > high_efficiency and e33 > high_efficiency:
        e66 = nnz / (36 * count_blocks(A, (6, 6)))
        if e66 > efficiency:
            return (6, 6)
        else:
            return (3, 3)
    else:
        if M % 4 == 0 and N % 4 == 0:
            e44 = nnz / (16 * count_blocks(A, (4, 4)))
        else:
            e44 = 0.0
        if e44 > efficiency:
            return (4, 4)
        elif e33 > efficiency:
            return (3, 3)
        elif e22 > efficiency:
            return (2, 2)
        else:
            return (1, 1)