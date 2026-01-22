import spherogram
import snappy
import numpy as np
import mpmath
from sage.all import PolynomialRing, LaurentPolynomialRing, RR, ZZ, RealField, ComplexField, matrix, arccos, exp
def basic_knot_test():
    for M in snappy.HTLinkExteriors(cusps=1):
        print(M.name())
        R = PolynomialRing(ZZ, 't')
        K = M.link()
        V = matrix(K.seifert_matrix())
        p0 = R(M.alexander_polynomial())
        p1 = R(K.alexander_polynomial())
        p2 = alexander_poly_from_seifert(V)
        assert p0 == p1 == p2
        partition, values = signature_function_of_integral_matrix(V)
        n = len(values)
        assert n % 2 == 1
        m = (n - 1) // 2
        assert K.signature() == values[m]