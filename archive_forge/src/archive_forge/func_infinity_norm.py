from ..sage_helper import _within_sage
from ..pari import pari, prec_dec_to_bits, prec_bits_to_dec, Gen
def infinity_norm(L):
    if is_pari(L):
        L = pari_vector_to_list(L)
    return max([abs(x) for x in L])