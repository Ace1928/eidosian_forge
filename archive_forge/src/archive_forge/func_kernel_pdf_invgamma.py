import numpy as np
from scipy import special, stats
def kernel_pdf_invgamma(x, sample, bw):
    return stats.invgamma.pdf(sample, 1 / bw + 1, scale=x / bw)