import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
def clone_with_theta(self, theta):
    cloned = clone(self)
    cloned.theta = theta
    return cloned