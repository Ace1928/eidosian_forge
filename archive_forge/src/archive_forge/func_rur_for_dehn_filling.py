import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def rur_for_dehn_filling(manifold):
    import giac_rur
    I = ptolemy_ideal_for_filled(manifold)
    return giac_rur.rational_univariate_representation(I)