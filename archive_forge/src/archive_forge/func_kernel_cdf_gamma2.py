import numpy as np
from scipy import special, stats
def kernel_cdf_gamma2(x, sample, bw):
    if np.size(x) == 1:
        if x < 2 * bw:
            a = (x / bw) ** 2 + 1
        else:
            a = x / bw
    else:
        a = x / bw
        mask = x < 2 * bw
        a[mask] = a[mask] ** 2 + 1
    pdf = stats.gamma.sf(sample, a, scale=bw)
    return pdf