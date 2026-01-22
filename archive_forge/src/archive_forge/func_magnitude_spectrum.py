import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.category  # Register category unit converter as side effect.
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates  # noqa # Register date unit converter as side effect.
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
@_preprocess_data(replace_names=['x'])
@_docstring.dedent_interpd
def magnitude_spectrum(self, x, Fs=None, Fc=None, window=None, pad_to=None, sides=None, scale=None, **kwargs):
    """
        Plot the magnitude spectrum.

        Compute the magnitude spectrum of *x*.  Data is padded to a
        length of *pad_to* and the windowing function *window* is applied to
        the signal.

        Parameters
        ----------
        x : 1-D array or sequence
            Array or sequence containing the data.

        %(Spectral)s

        %(Single_Spectrum)s

        scale : {'default', 'linear', 'dB'}
            The scaling of the values in the *spec*.  'linear' is no scaling.
            'dB' returns the values in dB scale, i.e., the dB amplitude
            (20 * log10). 'default' is 'linear'.

        Fc : int, default: 0
            The center frequency of *x*, which offsets the x extents of the
            plot to reflect the frequency range used when a signal is acquired
            and then filtered and downsampled to baseband.

        Returns
        -------
        spectrum : 1-D array
            The values for the magnitude spectrum before scaling (real valued).

        freqs : 1-D array
            The frequencies corresponding to the elements in *spectrum*.

        line : `~matplotlib.lines.Line2D`
            The line created by this function.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Keyword arguments control the `.Line2D` properties:

            %(Line2D:kwdoc)s

        See Also
        --------
        psd
            Plots the power spectral density.
        angle_spectrum
            Plots the angles of the corresponding frequencies.
        phase_spectrum
            Plots the phase (unwrapped angle) of the corresponding frequencies.
        specgram
            Can plot the magnitude spectrum of segments within the signal in a
            colormap.
        """
    if Fc is None:
        Fc = 0
    spec, freqs = mlab.magnitude_spectrum(x=x, Fs=Fs, window=window, pad_to=pad_to, sides=sides)
    freqs += Fc
    yunits = _api.check_getitem({None: 'energy', 'default': 'energy', 'linear': 'energy', 'dB': 'dB'}, scale=scale)
    if yunits == 'energy':
        Z = spec
    else:
        Z = 20.0 * np.log10(spec)
    line, = self.plot(freqs, Z, **kwargs)
    self.set_xlabel('Frequency')
    self.set_ylabel('Magnitude (%s)' % yunits)
    return (spec, freqs, line)