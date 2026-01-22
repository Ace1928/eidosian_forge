import numpy as np
def sigclip(self, sigs):
    """
        clips out all data points that are more than a certain number
        of standard deviations from the mean.

        sigs can be either a single value or a length-p sequence that
        specifies the number of standard deviations along each of the
        p dimensions.
        """
    if np.isscalar(sigs):
        sigs = sigs * np.ones(self.N.shape[1])
    sigs = sigs * np.std(self.N, axis=1)
    n = self.N.shape[0]
    m = np.all(np.abs(self.N) < sigs, axis=1)
    self.A = self.A[m]
    self.__calc()
    return n - sum(m)