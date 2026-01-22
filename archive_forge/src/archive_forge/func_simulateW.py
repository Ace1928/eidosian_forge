import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
def simulateW(self, nobs=100, T=1, dt=None, nrepl=1):
    """generate sample of Wiener Process
        """
    dt = T * 1.0 / nobs
    t = np.linspace(dt, 1, nobs)
    dW = np.sqrt(dt) * np.random.normal(size=(nrepl, nobs))
    W = np.cumsum(dW, 1)
    self.dW = dW
    return (W, t)