import numpy as np
def getCovarianceMatrix(self):
    """
        returns the covariance matrix for the dataset
        """
    return np.cov(self.N.T)