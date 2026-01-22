import mpmath
from mpmath import *
from mpmath.libmp import *
import random
import sys
def test_arithmetic_functions():
    import operator
    ops = [(operator.add, fadd), (operator.sub, fsub), (operator.mul, fmul), (operator.truediv, fdiv)]
    a = mpf(0.27)
    b = mpf(1.13)
    c = mpc(0.51 + 2.16j)
    d = mpc(1.08 - 0.99j)
    for x in [a, b, c, d]:
        for y in [a, b, c, d]:
            for op, fop in ops:
                if fop is not fdiv:
                    mp.prec = 200
                    z0 = op(x, y)
                mp.prec = 60
                z1 = op(x, y)
                mp.prec = 53
                z2 = op(x, y)
                assert fop(x, y, prec=60) == z1
                assert fop(x, y) == z2
                if fop is not fdiv:
                    assert fop(x, y, prec=inf) == z0
                    assert fop(x, y, dps=inf) == z0
                    assert fop(x, y, exact=True) == z0
                assert fneg(fneg(z1, exact=True), prec=inf) == z1
                assert fneg(z1) == -+z1
    mp.dps = 15