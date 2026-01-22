import numpy as np
from scipy import stats
from scipy.special import comb
from scipy.stats.distributions import rv_continuous
import matplotlib.pyplot as plt
from numpy import where, inf
from numpy import abs as np_abs
def meanexcess_plot(data, params=None, lidx=100, uidx=10, method='emp', plot=0):
    if method == 'est':
        if params is None:
            raise NotImplementedError
        else:
            pass
    elif method == 'emp':
        datasorted = np.sort(data)
        meanexcess = datasorted[::-1].cumsum() / np.arange(1, len(data) + 1) - datasorted[::-1]
        meanexcess = meanexcess[::-1]
        if plot:
            plt.plot(datasorted[:-uidx], meanexcess[:-uidx])
            if params is not None:
                shape, scale = params
                plt.plot(datasorted[:-uidx], (scale - datasorted[:-uidx] * shape) / (1.0 + shape))
    return (datasorted, meanexcess)