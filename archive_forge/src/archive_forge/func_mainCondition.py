from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def mainCondition(key_zij, key_zji, key_zkl, key_zlk):
    lhs = self[key_zij] * self[key_zji]
    rhs = (self[key_zkl] * self[key_zlk]).conj()
    if not is_zero(lhs - rhs):
        reason = '%s * %s = conjugate(%s * %s) not fulfilled' % (key_zij, key_zji, key_zkl, key_zlk)
        return NotPU21Representation(reason)
    return True