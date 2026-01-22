import numpy as np
import numpy.linalg as la
def kernel_vector(self, x, X, nsample):
    return np.hstack([self.kernel(x, x2) for x2 in X])