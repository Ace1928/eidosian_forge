from snappy.SnapPy import matrix, vector
from snappy.hyperboloid import (r13_dot,
def complex_and_height_to_R13_time_vector(z, t):
    """
    Takes a point in the upper half space model
    H^3 = { z + t * j : z in C, t > 0 } and gives the
    corresponding point in the 1,3-hyperboloid model.
    """
    z_re = z.real()
    z_im = z.imag()
    z_abs_sqr = z_re ** 2 + z_im ** 2
    denom = z_abs_sqr + (t + 1) ** 2
    poincare = [(z_abs_sqr + t ** 2 - 1) / denom, 2 * z_re / denom, 2 * z_im / denom]
    poincare_rsqr = sum([x ** 2 for x in poincare])
    klein_factor = 2 / (1 + poincare_rsqr)
    RF = z_re.parent()
    return R13_normalise(vector([RF(1), klein_factor * poincare[0], klein_factor * poincare[1], klein_factor * poincare[2]]))