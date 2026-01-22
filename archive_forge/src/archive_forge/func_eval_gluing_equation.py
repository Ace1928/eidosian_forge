from ..sage_helper import _within_sage
from ..pari import pari, prec_dec_to_bits, prec_bits_to_dec, Gen
def eval_gluing_equation(eqn, shapes):
    if is_pari(eqn):
        shapes = pari_vector_to_list(shapes)
    a, b, c = eqn
    ans = int(c)
    for i, z in enumerate(shapes):
        ans = ans * (z ** int(a[i]) * (1 - z) ** int(b[i]))
    return ans