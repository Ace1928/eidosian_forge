import math
from warnings import warn
import numpy as np
from numpy.linalg import inv
from scipy import optimize, spatial
def predict_xy(self, t, params=None):
    """Predict x- and y-coordinates using the estimated model.

        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (5,) array, optional
            Optional custom parameter set.

        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.

        """
    if params is None:
        params = self.params
    xc, yc, a, b, theta = params
    ct = np.cos(t)
    st = np.sin(t)
    ctheta = math.cos(theta)
    stheta = math.sin(theta)
    x = xc + a * ctheta * ct - b * stheta * st
    y = yc + a * stheta * ct + b * ctheta * st
    return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)