import sys, snappy, giac_rur, extended, phc_wrapper, time, gluing
from sage.all import QQ, PolynomialRing, CC, QQbar, macaulay2
def gluing_phc(manifold, vars_per_tet=2):
    I = gluing.gluing_variety_ideal(manifold, vars_per_tet)
    sols = phc_wrapper.phcpy_direct(I)
    output_vars = ['z%d' % i for i in range(manifold.num_tetrahedra())]
    return [sol for sol in sols if all((abs(sol[z] - 1) > 1e-07 for z in output_vars))]