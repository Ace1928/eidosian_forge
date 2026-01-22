from ..sage_helper import _within_sage
from ..pari import pari, prec_dec_to_bits, prec_bits_to_dec, Gen
def gluing_equation_errors(eqns, shapes):
    return [eval_gluing_equation(eqn, shapes) - 1 for eqn in eqns]