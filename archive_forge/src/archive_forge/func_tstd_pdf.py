import numpy as np
from scipy import special
from scipy.special import gammaln
def tstd_pdf(x, df):
    """pdf for standardized (not standard) t distribution, variance is one

    """
    r = np.array(df * 1.0)
    Px = np.exp(special.gammaln((r + 1) / 2.0) - special.gammaln(r / 2.0)) / np.sqrt((r - 2) * np.pi)
    Px /= (1 + x ** 2 / (r - 2)) ** ((r + 1) / 2.0)
    return Px