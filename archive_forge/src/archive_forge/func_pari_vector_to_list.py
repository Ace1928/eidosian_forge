from ..sage_helper import _within_sage
from ..pari import pari, prec_dec_to_bits, prec_bits_to_dec, Gen
def pari_vector_to_list(v):
    return v.Vec().python_list()