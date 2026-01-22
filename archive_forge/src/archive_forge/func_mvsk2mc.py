import numpy as np
from scipy.special import comb
def mvsk2mc(args):
    """convert mean, variance, skew, kurtosis to central moments"""
    X = _convert_to_multidim(args)

    def _local_counts(args):
        mu, sig2, sk, kur = args
        cnt = [None] * 4
        cnt[0] = mu
        cnt[1] = sig2
        cnt[2] = sk * sig2 ** 1.5
        cnt[3] = (kur + 3.0) * sig2 ** 2.0
        return tuple(cnt)
    res = np.apply_along_axis(_local_counts, 0, X)
    return _convert_from_multidim(res, tuple)