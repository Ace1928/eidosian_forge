import numpy as np
from scipy import special, stats
def kernel_cdf_bs(x, sample, bw):
    return stats.fatiguelife.sf(sample, bw, scale=x)