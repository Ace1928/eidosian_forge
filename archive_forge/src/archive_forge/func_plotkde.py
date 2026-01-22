import numpy as np
from numpy.testing import assert_almost_equal, assert_
import scipy
from scipy import stats
import matplotlib.pylab as plt
def plotkde(covfact):
    gkde.reset_covfact(covfact)
    kdepdf = gkde.evaluate(ind)
    plt.figure()
    plt.hist(xn, bins=20, normed=1)
    plt.plot(ind, kdepdf, label='kde', color='g')
    plt.plot(ind, alpha * stats.norm.pdf(ind, loc=mlow) + (1 - alpha) * stats.norm.pdf(ind, loc=mhigh), color='r', label='DGP: normal mix')
    plt.title('Kernel Density Estimation - ' + str(gkde.covfact))
    plt.legend()