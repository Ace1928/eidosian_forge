from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def detect_zero(polymod, exponent):
    if polymod == 0:
        if exponent < 0:
            raise Exception('RUR division by 0')
        return exponent > 0
    return False