import numpy as np
import numpy.linalg as la
def dK_dl_j(self, x1, x2):
    """Returns the derivative of the gradient of the kernel function
        respect to l
        """
    prefactor = -2 * (1 - 0.5 * self.squared_distance(x1, x2)) / self.l
    return self.kernel_function_gradient(x1, x2) * prefactor