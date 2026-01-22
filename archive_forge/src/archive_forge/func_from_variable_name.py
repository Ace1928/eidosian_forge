import re
import operator
from fractions import Fraction
import sys
@classmethod
def from_variable_name(cls, var):
    """Construct a polynomial consisting of a single variable."""
    return Polynomial((Monomial.from_variable_name(var),))