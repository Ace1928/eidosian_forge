from sympy.polys.galoistools import gf_from_dict, gf_factor_sqf
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
def gathen_poly(n, p, K):
    return gf_from_dict({n: K.one, 1: K.one, 0: K.one}, p, K)