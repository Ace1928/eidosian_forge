import numpy as np
def getEigensystem(self):
    """
        returns a tuple of (eigenvalues,eigenvectors) for the data set.
        """
    if self._eig is None:
        res = np.linalg.eig(self.getCovarianceMatrix())
        sorti = np.argsort(res[0])[::-1]
        res = (res[0][sorti], res[1][:, sorti])
        self._eig = res
    return self._eig