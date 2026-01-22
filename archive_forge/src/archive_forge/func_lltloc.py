import numpy as np
from scipy import special
from scipy.special import gammaln
def lltloc(loc, y, scale, df):
    return np.log(stats.t.pdf(y, df, loc=loc, scale=scale))