from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def latex_of_unit(quant):
    """Returns LaTeX representation of the unit of a quantity

    Examples
    --------
    >>> print(latex_of_unit(1/default_units.kelvin))
    \\mathrm{\\frac{1}{K}}

    """
    return _latex_from_dimensionality(quant.dimensionality).strip('$')