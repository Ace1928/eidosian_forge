import numpy as np
import scipy.linalg
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
def cauchy_point(self):
    """
        The Cauchy point is minimal along the direction of steepest descent.
        """
    if self._cauchy_point is None:
        g = self.jac
        Bg = self.hessp(g)
        self._cauchy_point = -(np.dot(g, g) / np.dot(g, Bg)) * g
    return self._cauchy_point