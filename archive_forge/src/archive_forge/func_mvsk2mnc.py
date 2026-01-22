import numpy as np
from scipy.special import comb
def mvsk2mnc(args):
    """convert mean, variance, skew, kurtosis to non-central moments"""
    X = _convert_to_multidim(args)

    def _local_counts(args):
        mc, mc2, skew, kurt = args
        mnc = mc
        mnc2 = mc2 + mc * mc
        mc3 = skew * mc2 ** 1.5
        mnc3 = mc3 + 3 * mc * mc2 + mc ** 3
        mc4 = (kurt + 3.0) * mc2 ** 2.0
        mnc4 = mc4 + 4 * mc * mc3 + 6 * mc * mc * mc2 + mc ** 4
        return (mnc, mnc2, mnc3, mnc4)
    res = np.apply_along_axis(_local_counts, 0, X)
    return _convert_from_multidim(res, tuple)