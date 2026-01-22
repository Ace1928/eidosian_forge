import numpy as np
from scipy import special, stats
def kernel_pdf_invgauss_(x, sample, bw):
    """Inverse gaussian kernel density, explicit formula.

    Scaillet 2004
    """
    pdf = 1 / np.sqrt(2 * np.pi * bw * sample ** 3) * np.exp(-1 / (2 * bw * x) * (sample / x - 2 + x / sample))
    return pdf.mean(-1)