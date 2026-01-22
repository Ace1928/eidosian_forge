import matplotlib.path as mpath
import numpy as np
def intersection_point(p0, p1, p2, p3):
    """
    Returns
    -------
    x, y
        The intersection point of the two infinite lines that pass through
        point p0->p1 and p2->p3 respectively.

    """
    x_1, y_1 = p0
    x_2, y_2 = p1
    x_3, y_3 = p2
    x_4, y_4 = p3
    div = (x_1 - x_2) * (y_3 - y_4) - (y_1 - y_2) * (x_3 - x_4)
    if div == 0:
        raise ValueError('Lines are parallel and cannot intersect at any one point.')
    x = ((x_1 * y_2 - y_1 * x_2) * (x_3 - x_4) - (x_1 - x_2) * (x_3 * y_4 - y_3 * x_4)) / div
    y = ((x_1 * y_2 - y_1 * x_2) * (y_3 - y_4) - (y_1 - y_2) * (x_3 * y_4 - y_3 * x_4)) / div
    return (x, y)