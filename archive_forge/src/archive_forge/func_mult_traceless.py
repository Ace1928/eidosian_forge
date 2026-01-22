from ..pari import pari
import string
from itertools import combinations, combinations_with_replacement, product
def mult_traceless(a1, a2, a3=None):
    """
    Takes 2 or 3 words and returns the trace of their product after
    making them traceless via M-> M-1/2tr(M)*I.
    """
    if a3 is None:
        return tr(a1 * a2) - pari('1/2') * tr(a1) * tr(a2)
    else:
        return pari('5/8') * tr(a1) * tr(a2) * tr(a3) - pari('1/2') * tr(a3) * tr(a1 * a2) - pari('1/2') * tr(a2) * tr(a1 * a3) - pari('1/2') * tr(a1) * tr(a2 * a3) + tr(a1 * a2 * a3)