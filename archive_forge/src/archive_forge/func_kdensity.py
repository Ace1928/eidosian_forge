import numpy as np
from scipy import integrate, stats
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.validation import array_like, float_like
from . import bandwidths
from .kdetools import forrt, revrt, silverman_transform
from .linbin import fast_linbin
def kdensity(x, kernel='gau', bw='normal_reference', weights=None, gridsize=None, adjust=1, clip=(-np.inf, np.inf), cut=3, retgrid=True):
    """
    Rosenblatt-Parzen univariate kernel density estimator.

    Parameters
    ----------
    x : array_like
        The variable for which the density estimate is desired.
    kernel : str
        The Kernel to be used. Choices are
        - "biw" for biweight
        - "cos" for cosine
        - "epa" for Epanechnikov
        - "gau" for Gaussian.
        - "tri" for triangular
        - "triw" for triweight
        - "uni" for uniform
    bw : str, float, callable
        The bandwidth to use. Choices are:

        - "scott" - 1.059 * A * nobs ** (-1/5.), where A is
          `min(std(x),IQR/1.34)`
        - "silverman" - .9 * A * nobs ** (-1/5.), where A is
          `min(std(x),IQR/1.34)`
        - "normal_reference" - C * A * nobs ** (-1/5.), where C is
          calculated from the kernel. Equivalent (up to 2 dp) to the
          "scott" bandwidth for gaussian kernels. See bandwidths.py
        - If a float is given, its value is used as the bandwidth.
        - If a callable is given, it's return value is used.
          The callable should take exactly two parameters, i.e.,
          fn(x, kern), and return a float, where:

          * x - the clipped input data
          * kern - the kernel instance used

    weights : array or None
        Optional  weights. If the x value is clipped, then this weight is
        also dropped.
    gridsize : int
        If gridsize is None, max(len(x), 50) is used.
    adjust : float
        An adjustment factor for the bw. Bandwidth becomes bw * adjust.
    clip : tuple
        Observations in x that are outside of the range given by clip are
        dropped. The number of observations in x is then shortened.
    cut : float
        Defines the length of the grid past the lowest and highest values of x
        so that the kernel goes to zero. The end points are
        -/+ cut*bw*{min(x) or max(x)}
    retgrid : bool
        Whether or not to return the grid over which the density is estimated.

    Returns
    -------
    density : ndarray
        The densities estimated at the grid points.
    grid : ndarray, optional
        The grid points at which the density is estimated.

    Notes
    -----
    Creates an intermediate (`gridsize` x `nobs`) array. Use FFT for a more
    computationally efficient version.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    clip_x = np.logical_and(x > clip[0], x < clip[1])
    x = x[clip_x]
    nobs = len(x)
    if gridsize is None:
        gridsize = max(nobs, 50)
    if weights is None:
        weights = np.ones(nobs)
        q = nobs
    else:
        weights = np.asarray(weights)
        if len(weights) != len(clip_x):
            msg = 'The length of the weights must be the same as the given x.'
            raise ValueError(msg)
        weights = weights[clip_x.squeeze()]
        q = weights.sum()
    kern = kernel_switch[kernel]()
    if callable(bw):
        bw = float(bw(x, kern))
    elif isinstance(bw, str):
        bw = bandwidths.select_bandwidth(x, bw, kern)
    else:
        bw = float_like(bw, 'bw')
    bw *= adjust
    a = np.min(x, axis=0) - cut * bw
    b = np.max(x, axis=0) + cut * bw
    grid = np.linspace(a, b, gridsize)
    k = (x.T - grid[:, None]) / bw
    kern.seth(bw)
    if kern.domain is not None:
        z_lo, z_high = kern.domain
        domain_mask = (k < z_lo) | (k > z_high)
        k = kern(k)
        k[domain_mask] = 0
    else:
        k = kern(k)
    k[k < 0] = 0
    dens = np.dot(k, weights) / (q * bw)
    if retgrid:
        return (dens, grid, bw)
    else:
        return (dens, bw)