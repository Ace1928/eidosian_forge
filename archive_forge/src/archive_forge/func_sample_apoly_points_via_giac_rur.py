import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def sample_apoly_points_via_giac_rur(manifold, n):
    import giac_rur
    I = extended_ptolemy_equations(manifold)
    R = I.ring()
    p = cyclotomic_polynomial(n, var=R('M'))
    I = I + [p]
    return giac_rur.rational_univariate_representation(I)