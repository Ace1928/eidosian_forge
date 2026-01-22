import numpy as np
from scipy import signal
from statsmodels.tsa.tsatools import lagmat
def padone(x, front=0, back=0, axis=0, fillvalue=0):
    """pad with zeros along one axis, currently only axis=0


    can be used sequentially to pad several axis

    Examples
    --------
    >>> padone(np.ones((2,3)),1,3,axis=1)
    array([[ 0.,  1.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.,  0.,  0.]])

    >>> padone(np.ones((2,3)),1,1, fillvalue=np.nan)
    array([[ NaN,  NaN,  NaN],
           [  1.,   1.,   1.],
           [  1.,   1.,   1.],
           [ NaN,  NaN,  NaN]])
    """
    shape = np.array(x.shape)
    shape[axis] += front + back
    shapearr = np.array(x.shape)
    out = np.empty(shape)
    out.fill(fillvalue)
    startind = np.zeros(x.ndim)
    startind[axis] = front
    endind = startind + shapearr
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    out[tuple(myslice)] = x
    return out