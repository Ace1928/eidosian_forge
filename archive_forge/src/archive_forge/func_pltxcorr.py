import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from statsmodels import regression
from statsmodels.tsa.arima_process import arma_generate_sample, arma_impulse_response
from statsmodels.tsa.arima_process import arma_acovf, arma_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf
def pltxcorr(self, x, y, normed=True, detrend=detrend_none, usevlines=True, maxlags=10, **kwargs):
    """
    call signature::

        def xcorr(self, x, y, normed=True, detrend=detrend_none,
          usevlines=True, maxlags=10, **kwargs):

    Plot the cross correlation between *x* and *y*.  If *normed* =
    *True*, normalize the data by the cross correlation at 0-th
    lag.  *x* and y are detrended by the *detrend* callable
    (default no normalization).  *x* and *y* must be equal length.

    Data are plotted as ``plot(lags, c, **kwargs)``

    Return value is a tuple (*lags*, *c*, *line*) where:

      - *lags* are a length ``2*maxlags+1`` lag vector

      - *c* is the ``2*maxlags+1`` auto correlation vector

      - *line* is a :class:`~matplotlib.lines.Line2D` instance
         returned by :func:`~matplotlib.pyplot.plot`.

    The default *linestyle* is *None* and the default *marker* is
    'o', though these can be overridden with keyword args.  The
    cross correlation is performed with :func:`numpy.correlate`
    with *mode* = 2.

    If *usevlines* is *True*:

       :func:`~matplotlib.pyplot.vlines`
       rather than :func:`~matplotlib.pyplot.plot` is used to draw
       vertical lines from the origin to the xcorr.  Otherwise the
       plotstyle is determined by the kwargs, which are
       :class:`~matplotlib.lines.Line2D` properties.

       The return value is a tuple (*lags*, *c*, *linecol*, *b*)
       where *linecol* is the
       :class:`matplotlib.collections.LineCollection` instance and
       *b* is the *x*-axis.

    *maxlags* is a positive integer detailing the number of lags to show.
    The default value of *None* will return all ``(2*len(x)-1)`` lags.

    **Example:**

    :func:`~matplotlib.pyplot.xcorr` above, and
    :func:`~matplotlib.pyplot.acorr` below.

    **Example:**

    .. plot:: mpl_examples/pylab_examples/xcorr_demo.py
    """
    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')
    x = detrend(np.asarray(x))
    y = detrend(np.asarray(y))
    c = np.correlate(x, y, mode=2)
    if normed:
        c /= np.sqrt(np.dot(x, x) * np.dot(y, y))
    if maxlags is None:
        maxlags = Nx - 1
    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maxlags must be None or strictly positive < %d' % Nx)
    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    if usevlines:
        a = self.vlines(lags, [0], c, **kwargs)
        b = self.axhline(**kwargs)
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', 'None')
        d = self.plot(lags, c, **kwargs)
    else:
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', 'None')
        a, = self.plot(lags, c, **kwargs)
        b = None
    return (lags, c, a, b)