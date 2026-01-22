import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def parse_ptolemy_edge(var):
    c, index, tet = var.split('_')
    tet = int(tet)
    edge = tuple((i for i in range(4) if index[i] == '1'))
    return (tet, edge)