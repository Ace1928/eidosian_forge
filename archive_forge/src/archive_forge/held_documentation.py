import numpy as np
from . import Filter  # prevent circular import in Python < 3.5
Design 2 filters with the Held construction (tight frame).

    This function create a parseval filterbank of :math:`2` filters.
    The low-pass filter is defined by the function

    .. math:: f_{l}=\begin{cases} 1 & \mbox{if }x\leq a\\
            \sin\left(2\pi\mu\left(\frac{x}{8a}\right)\right) & \mbox{if }a<x\leq2a\\
            0 & \mbox{if }x>2a \end{cases}

    with

    .. math:: \mu(x) = -1+24x-144*x^2+256*x^3

    The high pass filter is adaptated to obtain a tight frame.

    Parameters
    ----------
    G : graph
    a : float
        See equation above for this parameter
        The spectrum is scaled between 0 and 2 (default = 2/3)

    Examples
    --------

    Filter bank's representation in Fourier and time (ring graph) domains.

    >>> import matplotlib.pyplot as plt
    >>> G = graphs.Ring(N=20)
    >>> G.estimate_lmax()
    >>> G.set_coordinates('line1D')
    >>> g = filters.Held(G)
    >>> s = g.localize(G.N // 2)
    >>> fig, axes = plt.subplots(1, 2)
    >>> g.plot(ax=axes[0])
    >>> G.plot_signal(s, ax=axes[1])

    