import string
from ..sage_helper import _within_sage, sage_method
def clean_laurent(p, error):
    R = p.parent()
    t = R.gen()
    new_coeffs = [clean_CC(z, error) for z in p.coefficients()]
    return sum([a * t ** n for a, n in zip(new_coeffs, univ_exponents(p))], R.zero())