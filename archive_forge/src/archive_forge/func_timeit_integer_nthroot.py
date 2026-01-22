from sympy.core.numbers import Integer, Rational, integer_nthroot, igcd, pi, oo
from sympy.core.singleton import S
def timeit_integer_nthroot():
    integer_nthroot(100, 2)