import sys, os
from timeit import default_timer as clock
from mpmath import *
from mpmath.libmp.backend import exec_
def test_asymp(f, maxdps=150, verbose=False, huge_range=False):
    dps = [5, 15, 25, 50, 90, 150, 500, 1500, 5000, 10000]
    dps = [p for p in dps if p <= maxdps]

    def check(x, y, p, inpt):
        if abs(x - y) / abs(y) < workprec(20)(power)(10, -p + 1):
            return
        print()
        print('Error!')
        print('Input:', inpt)
        print('dps =', p)
        print('Result 1:', x)
        print('Result 2:', y)
        print('Absolute error:', abs(x - y))
        print('Relative error:', abs(x - y) / abs(y))
        raise AssertionError
    exponents = range(-20, 20)
    if huge_range:
        exponents += [-1000, -100, -50, 50, 100, 1000]
    for n in exponents:
        if verbose:
            sys.stdout.write('. ')
        mp.dps = 25
        xpos = mpf(10) ** n / 1.1287
        xneg = -xpos
        ximag = xpos * j
        xcomplex1 = xpos * (1 + j)
        xcomplex2 = xpos * (-1 + j)
        for i in range(len(dps)):
            if verbose:
                print('Testing dps = %s' % dps[i])
            mp.dps = dps[i]
            new = (f(xpos), f(xneg), f(ximag), f(xcomplex1), f(xcomplex2))
            if i != 0:
                p = dps[i - 1]
                check(prev[0], new[0], p, xpos)
                check(prev[1], new[1], p, xneg)
                check(prev[2], new[2], p, ximag)
                check(prev[3], new[3], p, xcomplex1)
                check(prev[4], new[4], p, xcomplex2)
            prev = new
    if verbose:
        print()