from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, curve_fit
from time import time
def asymptotic_series():
    """Asymptotic expansion for large x.

    Phi(a, b, x) ~ Z^(1/2-b) * exp((1+a)/a * Z) * sum_k (-1)^k * C_k / Z^k
    Z = (a*x)^(1/(1+a))

    Wright (1935) lists the coefficients C_0 and C_1 (he calls them a_0 and
    a_1). With slightly different notation, Paris (2017) lists coefficients
    c_k up to order k=3.
    Paris (2017) uses ZP = (1+a)/a * Z  (ZP = Z of Paris) and
    C_k = C_0 * (-a/(1+a))^k * c_k
    """
    order = 8

    class g(sympy.Function):
        """Helper function g according to Wright (1935)

        g(n, rho, v) = (1 + (rho+2)/3 * v + (rho+2)*(rho+3)/(2*3) * v^2 + ...)

        Note: Wright (1935) uses square root of above definition.
        """
        nargs = 3

        @classmethod
        def eval(cls, n, rho, v):
            if not n >= 0:
                raise ValueError('must have n >= 0')
            elif n == 0:
                return 1
            else:
                return g(n - 1, rho, v) + gammasimp(gamma(rho + 2 + n) / gamma(rho + 2)) / gammasimp(gamma(3 + n) / gamma(3)) * v ** n

    class coef_C(sympy.Function):
        """Calculate coefficients C_m for integer m.

        C_m is the coefficient of v^(2*m) in the Taylor expansion in v=0 of
        Gamma(m+1/2)/(2*pi) * (2/(rho+1))^(m+1/2) * (1-v)^(-b)
            * g(rho, v)^(-m-1/2)
        """
        nargs = 3

        @classmethod
        def eval(cls, m, rho, beta):
            if not m >= 0:
                raise ValueError('must have m >= 0')
            v = symbols('v')
            expression = (1 - v) ** (-beta) * g(2 * m, rho, v) ** (-m - Rational(1, 2))
            res = expression.diff(v, 2 * m).subs(v, 0) / factorial(2 * m)
            res = res * (gamma(m + Rational(1, 2)) / (2 * pi) * (2 / (rho + 1)) ** (m + Rational(1, 2)))
            return res
    xa, b, xap1 = symbols('xa b xap1')
    C0 = coef_C(0, xa, b)
    s = 'Asymptotic expansion for large x\n'
    s += 'Phi(a, b, x) = Z**(1/2-b) * exp((1+a)/a * Z) \n'
    s += '               * sum((-1)**k * C[k]/Z**k, k=0..6)\n\n'
    s += 'Z      = pow(a * x, 1/(1+a))\n'
    s += 'A[k]   = pow(a, k)\n'
    s += 'B[k]   = pow(b, k)\n'
    s += 'Ap1[k] = pow(1+a, k)\n\n'
    s += 'C[0] = 1./sqrt(2. * M_PI * Ap1[1])\n'
    for i in range(1, order + 1):
        expr = (coef_C(i, xa, b) / (C0 / (1 + xa) ** i)).simplify()
        factor = [x.denominator() for x in sympy.Poly(expr).coeffs()]
        factor = sympy.lcm(factor)
        expr = (expr * factor).simplify().collect(b, sympy.factor)
        expr = expr.xreplace({xa + 1: xap1})
        s += f'C[{i}] = C[0] / ({factor} * Ap1[{i}])\n'
        s += f'C[{i}] *= {str(expr)}\n\n'
    import re
    re_a = re.compile('xa\\*\\*(\\d+)')
    s = re_a.sub('A[\\1]', s)
    re_b = re.compile('b\\*\\*(\\d+)')
    s = re_b.sub('B[\\1]', s)
    s = s.replace('xap1', 'Ap1[1]')
    s = s.replace('xa', 'a')
    re_digits = re.compile('(\\d{10,})')
    s = re_digits.sub('\\1.', s)
    return s