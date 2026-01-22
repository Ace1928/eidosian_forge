from .z3 import *
from .z3core import *
from .z3printer import *
from fractions import Fraction
from .z3 import _get_ctx
def is_irrational(self):
    """ Return True if the numeral is irrational.

        >>> Numeral(2).is_irrational()
        False
        >>> Numeral("2/3").is_irrational()
        False
        >>> Numeral(Sqrt(2)).is_irrational()
        True
        """
    return not self.is_rational()