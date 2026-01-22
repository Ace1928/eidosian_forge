import numpy as np
from scipy import special, stats
def kernel_pdf_beta2(x, sample, bw):
    a1 = 2 * bw ** 2 + 2.5
    a2 = 4 * bw ** 4 + 6 * bw ** 2 + 2.25
    if np.size(x) == 1:
        if x < 2 * bw:
            a = a1 - np.sqrt(a2 - x ** 2 - x / bw)
            pdf = stats.beta.pdf(sample, a, (1 - x) / bw)
        elif x > 1 - 2 * bw:
            x_ = 1 - x
            a = a1 - np.sqrt(a2 - x_ ** 2 - x_ / bw)
            pdf = stats.beta.pdf(sample, x / bw, a)
        else:
            pdf = stats.beta.pdf(sample, x / bw, (1 - x) / bw)
    else:
        alpha = x / bw
        beta = (1 - x) / bw
        mask_low = x < 2 * bw
        x_ = x[mask_low]
        alpha[mask_low] = a1 - np.sqrt(a2 - x_ ** 2 - x_ / bw)
        mask_upp = x > 1 - 2 * bw
        x_ = 1 - x[mask_upp]
        beta[mask_upp] = a1 - np.sqrt(a2 - x_ ** 2 - x_ / bw)
        pdf = stats.beta.pdf(sample, alpha, beta)
    return pdf