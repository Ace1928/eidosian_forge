import numpy as np
from scipy import stats, signal
import matplotlib.pyplot as plt
def exactprocess(self, xzero, nobs, ddt=1.0, nrepl=2):
    """uses exact solution for log of process
        """
    lnxzero = np.log(xzero)
    lnx = super(self.__class__, self).exactprocess(xzero, nobs, ddt=ddt, nrepl=nrepl)
    return np.exp(lnx)