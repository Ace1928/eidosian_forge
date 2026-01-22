import numpy as np
def wrightomega_series_error(x):
    series = x
    desired = mpmath_wrightomega(x)
    return abs(series - desired) / desired