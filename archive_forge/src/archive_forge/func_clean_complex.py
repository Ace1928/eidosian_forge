import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def clean_complex(z, epsilon=1e-14):
    r, i = (abs(z.real), abs(z.imag))
    if r < epsilon and i < epsilon:
        return 0.0
    elif r < epsilon:
        ans = z.imag * 1j
    elif i < epsilon:
        ans = z.real
    else:
        ans = z
    assert abs(z - ans) < epsilon
    return ans