import contextlib
from collections import namedtuple
import datetime
from decimal import Decimal
from functools import partial
import inspect
import io
from itertools import product
import platform
from types import SimpleNamespace
import dateutil.tz
import numpy as np
from numpy import ma
from cycler import cycler
import pytest
import matplotlib
import matplotlib as mpl
from matplotlib import rc_context, patheffects
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.font_manager as mfont_manager
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.projections.geo import HammerAxes
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import mpl_toolkits.axisartist as AA  # type: ignore
from numpy.testing import (
from matplotlib.testing.decorators import (
@image_comparison(['magnitude_spectrum_freqs_linear.png', 'magnitude_spectrum_freqs_dB.png', 'angle_spectrum_freqs.png', 'phase_spectrum_freqs.png', 'magnitude_spectrum_noise_linear.png', 'magnitude_spectrum_noise_dB.png', 'angle_spectrum_noise.png', 'phase_spectrum_noise.png'], remove_text=True)
def test_spectrum():
    n = 10000
    Fs = 100.0
    fstims1 = [Fs / 4, Fs / 5, Fs / 11]
    NFFT = int(1000 * Fs / min(fstims1))
    pad_to = int(2 ** np.ceil(np.log2(NFFT)))
    x = np.arange(0, n, 1 / Fs)
    y_freqs = (np.sin(2 * np.pi * np.outer(x, fstims1)) * 10 ** np.arange(3)).sum(axis=1)
    np.random.seed(0)
    y_noise = np.hstack([np.random.standard_normal(n), np.random.rand(n)]) - 0.5
    all_sides = ['default', 'onesided', 'twosided']
    kwargs = {'Fs': Fs, 'pad_to': pad_to}
    for y in [y_freqs, y_noise]:
        for ax, sides in zip(plt.figure().subplots(3), all_sides):
            spec, freqs, line = ax.magnitude_spectrum(y, sides=sides, **kwargs)
            ax.set(xlabel='', ylabel='')
        for ax, sides in zip(plt.figure().subplots(3), all_sides):
            spec, freqs, line = ax.magnitude_spectrum(y, sides=sides, **kwargs, scale='dB')
            ax.set(xlabel='', ylabel='')
        for ax, sides in zip(plt.figure().subplots(3), all_sides):
            spec, freqs, line = ax.angle_spectrum(y, sides=sides, **kwargs)
            ax.set(xlabel='', ylabel='')
        for ax, sides in zip(plt.figure().subplots(3), all_sides):
            spec, freqs, line = ax.phase_spectrum(y, sides=sides, **kwargs)
            ax.set(xlabel='', ylabel='')