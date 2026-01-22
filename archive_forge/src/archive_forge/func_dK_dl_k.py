import numpy as np
import numpy.linalg as la
def dK_dl_k(self, x1, x2):
    """Returns the derivative of the kernel function respect to l"""
    return self.squared_distance(x1, x2) / self.l