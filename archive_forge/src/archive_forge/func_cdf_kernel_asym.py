import numpy as np
from scipy import special, stats
def cdf_kernel_asym(x, sample, bw, kernel_type, weights=None, batch_size=10):
    """Estimate of cumulative distribution based on asymmetric kernel.

    Parameters
    ----------
    x : array_like, float
        Points for which density is evaluated. ``x`` can be scalar or 1-dim.
    sample : ndarray, 1-d
        Sample from which kernel estimate is computed.
    bw : float
        Bandwidth parameter, there is currently no default value for it.
    kernel_type : str or callable
        Kernel name or kernel function.
        Currently supported kernel names are "beta", "beta2", "gamma",
        "gamma2", "bs", "invgamma", "invgauss", "lognorm", "recipinvgauss" and
        "weibull".
    weights : None or ndarray
        If weights is not None, then kernel for sample points are weighted
        by it. No weights corresponds to uniform weighting of each component
        with 1 / nobs, where nobs is the size of `sample`.
    batch_size : float
        If x is an 1-dim array, then points can be evaluated in vectorized
        form. To limit the amount of memory, a loop can work in batches.
        The number of batches is determined so that the intermediate array
        sizes are limited by

        ``np.size(batch) * len(sample) < batch_size * 1000``.

        Default is to have at most 10000 elements in intermediate arrays.

    Returns
    -------
    cdf : float or ndarray
        Estimate of cdf at points x. ``cdf`` has the same size or shape as x.
    """
    if callable(kernel_type):
        kfunc = kernel_type
    else:
        kfunc = kernel_dict_cdf[kernel_type]
    batch_size = batch_size * 1000
    if np.size(x) * len(sample) < batch_size:
        if np.size(x) > 1:
            x = np.asarray(x)[:, None]
        cdfi = kfunc(x, sample, bw)
        if weights is None:
            cdf = cdfi.mean(-1)
        else:
            cdf = cdfi @ weights
    else:
        if weights is None:
            weights = np.ones(len(sample)) / len(sample)
        k = batch_size // len(sample)
        n = len(x) // k
        x_split = np.array_split(x, n)
        cdf = np.concatenate([kfunc(xi[:, None], sample, bw) @ weights for xi in x_split])
    return cdf