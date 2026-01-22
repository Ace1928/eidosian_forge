import numpy as np
def deproject(self, A, normed=True):
    """
        input is an n X q array, where q <= p

        output is p X n
        """
    A = np.atleast_2d(A)
    n, q = A.shape
    p = self.A.shape[1]
    if q > p:
        raise ValueError('q > p')
    evinv = np.linalg.inv(np.matrix(self.getEigenvectors()).T)
    zs = np.zeros((n, p))
    zs[:, :q] = A
    proj = evinv * zs.T
    if normed:
        return np.array(proj.T).T
    else:
        mns = np.mean(self.A, axis=0)
        sds = np.std(self.M, axis=0)
        return (np.array(proj.T) * sds + mns).T