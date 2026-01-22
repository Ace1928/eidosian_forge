import numpy as np
from .version_requirements import require
def polygon_area(pr, pc):
    """Compute the area of a polygon.

    Parameters
    ----------
    pr, pc : (K,) array of float
        Polygon row and column coordinates.

    Returns
    -------
    a : float
        Area of the polygon.
    """
    pr = np.asarray(pr)
    pc = np.asarray(pc)
    return 0.5 * np.abs(np.sum(pc[:-1] * pr[1:] - pc[1:] * pr[:-1]))