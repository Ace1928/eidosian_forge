import numpy as np
from pygsp import utils
from . import approximations
def can_dual(self):
    """Creates a dual graph form a given graph"""

    def can_dual_func(g, n, x):
        x = np.ravel(x)
        N = np.shape(x)[0]
        M = g.Nf
        gcoeff = g.evaluate(x).T
        s = np.zeros((N, M))
        for i in range(N):
            s[i] = np.linalg.pinv(np.expand_dims(gcoeff[i], axis=1))
        ret = s[:, n]
        return ret
    kernels = []
    for i in range(self.Nf):
        kernels.append(lambda x, i=i: can_dual_func(self, i, x))
    return Filter(self.G, kernels)