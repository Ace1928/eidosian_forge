from math import cos, exp, sin, sqrt, atan2
def dist_rgb(rgb1, rgb2):
    """
    Determine distance between two rgb colors.

    :arg tuple rgb1: RGB color definition
    :arg tuple rgb2: RGB color definition
    :returns: Square of the distance between provided colors
    :rtype: float

    This works by treating RGB colors as coordinates in three dimensional
    space and finding the closest point within the configured color range
    using the formula::

        d^2 = (r2 - r1)^2 + (g2 - g1)^2 + (b2 - b1)^2

    For efficiency, the square of the distance is returned
    which is sufficient for comparisons
    """
    return sum((pow(rgb1[idx] - rgb2[idx], 2) for idx in (0, 1, 2)))