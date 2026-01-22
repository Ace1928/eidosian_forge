import math
from warnings import warn
import numpy as np
from numpy.linalg import inv
from scipy import optimize, spatial
def predict_y(self, x, params=None):
    """Predict y-coordinates for 2D lines using the estimated model.

        Alias for::

            predict(x, axis=0)[:, 1]

        Parameters
        ----------
        x : array
            x-coordinates.
        params : (2,) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        y : array
            Predicted y-coordinates.

        """
    y = self.predict(x, axis=0, params=params)[:, 1]
    return y