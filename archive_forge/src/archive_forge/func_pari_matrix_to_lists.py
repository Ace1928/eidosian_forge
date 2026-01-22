from ..sage_helper import _within_sage
from ..pari import pari, prec_dec_to_bits, prec_bits_to_dec, Gen
def pari_matrix_to_lists(A):
    """Return the entries of A in *column major* order"""
    return [pari_vector_to_list(v) for v in A.list()]