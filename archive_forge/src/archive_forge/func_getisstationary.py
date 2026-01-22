import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
def getisstationary(self, a=None):
    """check whether the auto-regressive lag-polynomial is stationary

        Returns
        -------
        isstationary : bool

        *attaches*

        areigenvalues : complex array
            eigenvalues sorted by absolute value

        References
        ----------
        formula taken from NAG manual

        """
    if a is not None:
        a = a
    elif self.isstructured:
        a = -self.reduceform(self.ar)[1:]
    else:
        a = -self.ar[1:]
    amat = self.stacksquare(a)
    ev = np.sort(np.linalg.eigvals(amat))[::-1]
    self.areigenvalues = ev
    return (np.abs(ev) < 1).all()