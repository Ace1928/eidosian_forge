import re
import operator
from fractions import Fraction
import sys
def parenthesis_coefficient_method(i):
    if isinstance(i, int) or isinstance(i, Fraction):
        return default_print_coefficient_method(i)
    else:
        return ('+', '(%s)' % repr(i))