import numpy as np
from scipy import special, stats
def kernel_pdf_beta(x, sample, bw):
    return stats.beta.pdf(sample, x / bw + 1, (1 - x) / bw + 1)