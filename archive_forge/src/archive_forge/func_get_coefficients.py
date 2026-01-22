import re
import operator
from fractions import Fraction
import sys
def get_coefficients(self, conversion_function=lambda x: x):
    """Assert univariance; return the coefficients in degree order."""
    assert self.is_univariate()
    degree = self.degree()
    list_of_coefficients = (degree + 1) * [conversion_function(0)]
    for monomial in self._monomials:
        list_of_coefficients[degree - monomial.degree()] = conversion_function(monomial.get_coefficient())
    return list_of_coefficients