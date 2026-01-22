import math
from warnings import warn
import numpy as np
from numpy.linalg import inv
from scipy import optimize, spatial
def predict_x(self, y, params=None):
    """Predict x-coordinates for 2D lines using the estimated model.

        Alias for::

            predict(y, axis=1)[:, 0]

        Parameters
        ----------
        y : array
            y-coordinates.
        params : (2,) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        x : array
            Predicted x-coordinates.

        """
    x = self.predict(y, axis=1, params=params)[:, 0]
    return x