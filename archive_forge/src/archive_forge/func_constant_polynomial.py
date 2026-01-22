import re
import operator
from fractions import Fraction
import sys
@classmethod
def constant_polynomial(cls, constant):
    """Construct a constant polynomial."""
    return Polynomial((Monomial.constant_monomial(constant),))