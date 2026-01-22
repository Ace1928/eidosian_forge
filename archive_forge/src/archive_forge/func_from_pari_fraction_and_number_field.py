from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
@staticmethod
def from_pari_fraction_and_number_field(fraction, poly):
    if poly:
        return RUR([(fraction.numerator().Mod(poly), 1), (fraction.denominator().Mod(poly), -1)])
    else:
        return RUR([(fraction.numerator(), 1), (fraction.denominator(), -1)])