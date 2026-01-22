from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def poly_involution(f):
    """
       sage: K = CyclotomicField(3, 'z')
       sage: R = PolynomialRing(K, 't')
       sage: z, t = K.gen(), R.gen()
       sage: poly_involution(z*t**2 + (1/z)*t + 1)
       t^2 + z*t - z - 1
    """
    R = f.parent()
    K = R.base_ring()
    z, t = (K.gen(), R.gen())
    bar = K.hom([1 / z])
    ans = R(0)
    d = f.degree()
    for e in f.exponents():
        ans += bar(f[e]) * t ** (d - e)
    return ans