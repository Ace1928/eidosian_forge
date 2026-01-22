from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, curve_fit
from time import time
def series_small_a():
    """Tylor series expansion of Phi(a, b, x) in a=0 up to order 5.
    """
    order = 5
    a, b, x, k = symbols('a b x k')
    A = []
    X = []
    B = []
    expression = Sum(x ** k / factorial(k) / gamma(a * k + b), (k, 0, S.Infinity))
    expression = gamma(b) / sympy.exp(x) * expression
    for n in range(0, order + 1):
        term = expression.diff(a, n).subs(a, 0).simplify().doit()
        x_part = term.subs(polygamma(0, b), 1).replace(polygamma, lambda *args: 0)
        x_part *= (-1) ** n
        A.append(a ** n / factorial(n))
        X.append(horner(x_part))
        B.append(horner((term / x_part).simplify()))
    s = 'Tylor series expansion of Phi(a, b, x) in a=0 up to order 5.\n'
    s += 'Phi(a, b, x) = exp(x)/gamma(b) * sum(A[i] * X[i] * B[i], i=0..5)\n'
    for name, c in zip(['A', 'X', 'B'], [A, X, B]):
        for i in range(len(c)):
            s += f'\n{name}[{i}] = ' + str(c[i])
    return s