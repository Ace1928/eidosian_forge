import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from statsmodels import regression
from statsmodels.tsa.arima_process import arma_generate_sample, arma_impulse_response
from statsmodels.tsa.arima_process import arma_acovf, arma_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, acovf
from statsmodels.graphics.tsaplots import plot_acf
def pltacorr(self, x, **kwargs):
    """
    call signature::

        acorr(x, normed=True, detrend=detrend_none, usevlines=True,
              maxlags=10, **kwargs)

    Plot the autocorrelation of *x*.  If *normed* = *True*,
    normalize the data by the autocorrelation at 0-th lag.  *x* is
    detrended by the *detrend* callable (default no normalization).

    Data are plotted as ``plot(lags, c, **kwargs)``

    Return value is a tuple (*lags*, *c*, *line*) where:

      - *lags* are a length 2*maxlags+1 lag vector

      - *c* is the 2*maxlags+1 auto correlation vector

      - *line* is a :class:`~matplotlib.lines.Line2D` instance
        returned by :meth:`plot`

    The default *linestyle* is None and the default *marker* is
    ``'o'``, though these can be overridden with keyword args.
    The cross correlation is performed with
    :func:`numpy.correlate` with *mode* = 2.

    If *usevlines* is *True*, :meth:`~matplotlib.axes.Axes.vlines`
    rather than :meth:`~matplotlib.axes.Axes.plot` is used to draw
    vertical lines from the origin to the acorr.  Otherwise, the
    plot style is determined by the kwargs, which are
    :class:`~matplotlib.lines.Line2D` properties.

    *maxlags* is a positive integer detailing the number of lags
    to show.  The default value of *None* will return all
    :math:`2 \\mathrm{len}(x) - 1` lags.

    The return value is a tuple (*lags*, *c*, *linecol*, *b*)
    where

    - *linecol* is the
      :class:`~matplotlib.collections.LineCollection`

    - *b* is the *x*-axis.

    .. seealso::

        :meth:`~matplotlib.axes.Axes.plot` or
        :meth:`~matplotlib.axes.Axes.vlines`
           For documentation on valid kwargs.

    **Example:**

    :func:`~matplotlib.pyplot.xcorr` above, and
    :func:`~matplotlib.pyplot.acorr` below.

    **Example:**

    .. plot:: mpl_examples/pylab_examples/xcorr_demo.py
    """
    return self.xcorr(x, x, **kwargs)