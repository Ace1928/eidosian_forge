import numpy as np
def to_restricted(p, sel, bounds):
    """
    Transform parameters from the unrestricted [0,1] space
    to satisfy both the bounds and the 2 constraints
    beta <= alpha and gamma <= (1-alpha)

    Parameters
    ----------
    p : ndarray
        The parameters to transform
    sel : ndarray
        Array indicating whether a parameter is being estimated. If not
        estimated, not transformed.
    bounds : ndarray
        2-d array of bounds where bound for element i is in row i
        and stored as [lb, ub]

    Returns
    -------

    """
    a, b, g = p[:3]
    if sel[0]:
        lb = max(LOWER_BOUND, bounds[0, 0])
        ub = min(1 - LOWER_BOUND, bounds[0, 1])
        a = lb + a * (ub - lb)
    if sel[1]:
        lb = bounds[1, 0]
        ub = min(a, bounds[1, 1])
        b = lb + b * (ub - lb)
    if sel[2]:
        lb = bounds[2, 0]
        ub = min(1.0 - a, bounds[2, 1])
        g = lb + g * (ub - lb)
    return (a, b, g)