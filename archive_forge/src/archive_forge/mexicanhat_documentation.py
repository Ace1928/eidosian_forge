from __future__ import division
import numpy as np
from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5
Design a filter bank of Mexican hat wavelets.

    The Mexican hat wavelet is the second oder derivative of a Gaussian. Since
    we express the filter in the Fourier domain, we find:

    .. math:: \hat{g}_b(x) = x * \exp(-x)

    for the band-pass filter. Note that in our convention the eigenvalues of
    the Laplacian are equivalent to the square of graph frequencies,
    i.e. :math:`x = \lambda^2`.

    The low-pass filter is given by

    .. math: \hat{g}_l(x) = \exp(-x^4).

    Parameters
    ----------
    G : graph
    Nf : int
        Number of filters to cover the interval [0, lmax].
    lpfactor : int
        Low-pass factor. lmin=lmax/lpfactor will be used to determine scales.
        The scaling function will be created to fill the low-pass gap.
    scales : array-like
        Scales to be used.
        By default, initialized with :func:`pygsp.utils.compute_log_scales`.
    normalize : bool
        Whether to normalize the wavelet by the factor ``sqrt(scales)``.

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.MexicanHat(G)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> g.plot(ax=axes[0])
    >>> G.plot_signal(s, ax=axes[1])

    