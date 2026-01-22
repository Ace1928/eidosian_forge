import string
from ..sage_helper import _within_sage, sage_method
def fox_derivative_with_involution(word, phi, var):
    """
    The group ring Z[pi] has a natural involution iota sends
    g in pi to g^-1 and respects addition.  This function
    computes

        phi( iota( d word / d var) )
    """
    R, phi_ims, fox_ders = setup_fox_derivative(word, phi, var, involute=True)
    D = 0
    for w in reverse_word(word):
        D = fox_ders[w] + D * phi_ims[w]
    return D