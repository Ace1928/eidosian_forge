import sys, snappy, giac_rur, extended, phc_wrapper, time, gluing
from sage.all import QQ, PolynomialRing, CC, QQbar, macaulay2
def ptolemy_nag4m2(manifold):
    """
    50-100 times slower than PHCpack even on the simplest examples.
    """
    I = extended.ptolemy_ideal_for_filled(manifold)
    macaulay2('loadPackage "NumericalAlgebraicGeometry"')
    macaulay2(I.ring())
    return macaulay2(I.gens()).toList().solveSystem()