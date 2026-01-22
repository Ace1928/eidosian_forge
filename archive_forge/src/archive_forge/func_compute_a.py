import os
from scipy.special._precompute.utils import lagrange_inversion
def compute_a(n):
    """a_k from DLMF 5.11.6"""
    a = [mp.sqrt(2) / 2]
    for k in range(1, n):
        ak = a[-1] / k
        for j in range(1, len(a)):
            ak -= a[j] * a[-j] / (j + 1)
        ak /= a[0] * (1 + mp.mpf(1) / (k + 1))
        a.append(ak)
    return a