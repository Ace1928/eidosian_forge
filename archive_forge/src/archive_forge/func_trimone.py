import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
def trimone(x, front=0, back=0, axis=0):
    """trim number of array elements along one axis


    Examples
    --------
    >>> xp = padone(np.ones((2,3)),1,3,axis=1)
    >>> xp
    array([[ 0.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.,  0.,  0.]])
    >>> trimone(xp,1,3,1)
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]])
    """
    shape = np.array(x.shape)
    shape[axis] -= front + back
    shapearr = np.array(x.shape)
    startind = np.zeros(x.ndim)
    startind[axis] = front
    endind = startind + shape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return x[tuple(myslice)]