import numpy as np
def prior(self, x):
    """Actual prior function, common to all Priors"""
    if len(x.shape) > 1:
        n = x.shape[0]
        return np.hstack([self.potential(x[i, :]) for i in range(n)])
    else:
        return self.potential(x)