import sys, snappy, giac_rur, extended, phc_wrapper, time, gluing
from sage.all import QQ, PolynomialRing, CC, QQbar, macaulay2
def ptolemy_phc_as_used(manifold):
    return extended.shapes_of_SL2C_reps_for_filled(manifold, phc_wrapper.phcpy_direct)