import numpy as np
def wrightomega_exp_error(x):
    exponential_approx = mpmath.exp(x)
    desired = mpmath_wrightomega(x)
    return abs(exponential_approx - desired) / desired