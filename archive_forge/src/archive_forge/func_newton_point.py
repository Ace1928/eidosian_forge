import numpy as np
import scipy.linalg
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)
def newton_point(self):
    """
        The Newton point is a global minimum of the approximate function.
        """
    if self._newton_point is None:
        g = self.jac
        B = self.hess
        cho_info = scipy.linalg.cho_factor(B)
        self._newton_point = -scipy.linalg.cho_solve(cho_info, g)
    return self._newton_point