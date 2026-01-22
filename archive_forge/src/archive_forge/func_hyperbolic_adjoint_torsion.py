import string
from ..sage_helper import _within_sage, sage_method
@sage_method
def hyperbolic_adjoint_torsion(manifold, bits_prec=100):
    """
    Computes the torsion polynomial of the adjoint representation
    a la Dubois-Yamaguichi.   This is not a sign-refined computation
    so the result is only defined up to sign, not to mention a power
    of the variable 'a'::

        sage: M = Manifold('K11n42')
        sage: tau = M.hyperbolic_adjoint_torsion()
        sage: tau.parent()
        Univariate Polynomial Ring in a over Complex Field with 100 bits of precision
        sage: tau.degree()
        7
    """
    return hyperbolic_SLN_torsion(manifold, 3, bits_prec)