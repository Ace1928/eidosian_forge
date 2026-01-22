from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def unicode_of_unit(quant):
    """Returns unicode representation of the unit of a quantity

    Examples
    --------
    >>> print(unicode_of_unit(1/default_units.kelvin))
    1/K

    """
    return quant.dimensionality.unicode