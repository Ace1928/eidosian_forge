from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def multiply_and_simplify_terms(self):
    """
        Multiplies all terms that make up the numerator and denominator
        and also reduce them. Reducing requires comupting the gcd.

        >>> nf = pari("x^97+x^3+x+32121")
        >>> a = pari("43*x^3 + 1")
        >>> b = pari("43*x^3 + 2")
        >>> c = pari("43*x^3 + 3")
        >>> d = pari("43*x^3 + 4")

        >>> r1 = RUR.from_pari_fraction_and_number_field(a * b / c, nf)
        >>> r2 = RUR.from_pari_fraction_and_number_field(c / a, nf)
        >>> r3 = RUR.from_pari_fraction_and_number_field(d, nf)
        >>> r = r1 * r2 * r3

        The c's cancel when multiplying.

        >>> r
        ( Mod(1849*x^6 + 129*x^3 + 2, x^97 + x^3 + x + 32121) * Mod(43*x^3 + 4, x^97 + x^3 + x + 32121) ) / ( Mod(43*x^3 + 1, x^97 + x^3 + x + 32121) )

        The terms in a * b and d in numerator get multiplied.

        >>> r.multiply_terms()
        ( Mod(79507*x^9 + 12943*x^6 + 602*x^3 + 8, x^97 + x^3 + x + 32121) ) / ( Mod(43*x^3 + 1, x^97 + x^3 + x + 32121) )

        Now the c's cancel as well.

        >>> r.multiply_and_simplify_terms()
        ( Mod(1849*x^6 + 258*x^3 + 8, x^97 + x^3 + x + 32121) )

        """
    return RUR.from_pari_fraction_and_number_field(self.to_pari_fraction(), self.number_field())