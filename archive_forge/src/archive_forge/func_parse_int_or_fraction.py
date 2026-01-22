import re
import operator
from fractions import Fraction
import sys
def parse_int_or_fraction(s):
    m = re.match('([0-9]+/[0-9]+)(.*)', s)
    if m:
        frac, rest = m.groups()
        return (Fraction(frac), rest)
    return parse_int_coefficient(s)