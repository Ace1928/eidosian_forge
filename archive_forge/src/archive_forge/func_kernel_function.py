import numpy as np
import numpy.linalg as la
def kernel_function(self, x1, x2):
    """ This is the squared exponential function"""
    return self.weight ** 2 * np.exp(-0.5 * self.squared_distance(x1, x2))