from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def process_pair(polymodExponent):
    polymod, exponent = polymodExponent
    if abs(exponent) == 1:
        return '%s' % polymod
    else:
        return '%s^%d' % (polymod, exponent)