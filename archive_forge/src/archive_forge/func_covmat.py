import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
def covmat(self):
    """ covariance matrix of estimate
        # not sure it's correct, need to check orientation everywhere
        # looks ok, display needs getting used to
        >>> v.rss[None,None,:]*np.linalg.inv(np.dot(v.xred.T,v.xred))[:,:,None]
        array([[[ 0.37247445,  0.32210609],
                [ 0.1002642 ,  0.08670584]],

               [[ 0.1002642 ,  0.08670584],
                [ 0.45903637,  0.39696255]]])
        >>>
        >>> v.rss[0]*np.linalg.inv(np.dot(v.xred.T,v.xred))
        array([[ 0.37247445,  0.1002642 ],
               [ 0.1002642 ,  0.45903637]])
        >>> v.rss[1]*np.linalg.inv(np.dot(v.xred.T,v.xred))
        array([[ 0.32210609,  0.08670584],
               [ 0.08670584,  0.39696255]])
       """
    self.paramcov = self.rss[None, None, :] * np.linalg.inv(np.dot(self.xred.T, self.xred))[:, :, None]