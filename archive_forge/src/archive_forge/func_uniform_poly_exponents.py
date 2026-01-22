import string
from ..sage_helper import _within_sage, sage_method
def uniform_poly_exponents(poly):
    if poly.parent().ngens() == 1:
        return [(e,) for e in poly.exponents()]
    return poly.exponents()