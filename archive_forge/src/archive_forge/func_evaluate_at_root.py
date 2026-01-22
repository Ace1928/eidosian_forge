from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def evaluate_at_root(self, root):
    """
        Given a numerical value for a root of the polynomial returned
        by number_field, evaluates all polynomials at that root and
        computes the value of the fraction.

        >>> nf = pari('x^2+x+1')
        >>> a = pari('x+2')
        >>> b = pari('3*x + 1/5')

        >>> root = pari('-0.5 + 0.8660254')

        >>> prev = pari.set_real_precision(15)

        >>> r = RUR.from_pari_fraction_and_number_field(a/b, nf)
        >>> r
        ( Mod(5*x + 10, x^2 + x + 1) ) / ( Mod(15*x + 1, x^2 + x + 1) )

        >>> r.evaluate_at_root(root)
        1.82271687902451

        >>> dummy = pari.set_real_precision(prev)

        """

    def evaluate_poly(p):
        if type(p) == Gen and p.type() == 't_POLMOD':
            return p.lift().substpol('x', root)
        return pari(p)
    return prod([evaluate_poly(p) ** e for p, e in self._polymod_exponent_pairs])