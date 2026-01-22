import numpy as np
from scipy import special, stats
def kernel_pdf_lognorm_(x, sample, bw):
    """Log-normal kernel for density, pdf, estimation, explicit formula.

    Jin, Kawczak 2003
    """
    term = 8 * np.log(1 + bw)
    pdf = 1 / np.sqrt(term * np.pi) / sample * np.exp(-(np.log(x) - np.log(sample)) ** 2 / term)
    return pdf.mean(-1)