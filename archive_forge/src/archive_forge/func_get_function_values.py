import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
def get_function_values(self, alpha, ecc, dofs):
    """
        Parameters
        ----------
        alpha : is a (N x 3 x 1) array (array of column-matrices) of
        barycentric coordinates,
        ecc : is a (N x 3 x 1) array (array of column-matrices) of triangle
        eccentricities,
        dofs : is a (N x 1 x 9) arrays (arrays of row-matrices) of computed
        degrees of freedom.

        Returns
        -------
        Returns the N-array of interpolated function values.
        """
    subtri = np.argmin(alpha, axis=1)[:, 0]
    ksi = _roll_vectorized(alpha, -subtri, axis=0)
    E = _roll_vectorized(ecc, -subtri, axis=0)
    x = ksi[:, 0, 0]
    y = ksi[:, 1, 0]
    z = ksi[:, 2, 0]
    x_sq = x * x
    y_sq = y * y
    z_sq = z * z
    V = _to_matrix_vectorized([[x_sq * x], [y_sq * y], [z_sq * z], [x_sq * z], [x_sq * y], [y_sq * x], [y_sq * z], [z_sq * y], [z_sq * x], [x * y * z]])
    prod = self.M @ V
    prod += _scalar_vectorized(E[:, 0, 0], self.M0 @ V)
    prod += _scalar_vectorized(E[:, 1, 0], self.M1 @ V)
    prod += _scalar_vectorized(E[:, 2, 0], self.M2 @ V)
    s = _roll_vectorized(prod, 3 * subtri, axis=0)
    return (dofs @ s)[:, 0, 0]