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
@image_comparison(['psd_freqs.png', 'csd_freqs.png', 'psd_noise.png', 'csd_noise.png'], remove_text=True, tol=0.002)
def test_psd_csd():
    n = 10000
    Fs = 100.0
    fstims = [[Fs / 4, Fs / 5, Fs / 11], [Fs / 4.7, Fs / 5.6, Fs / 11.9]]
    NFFT_freqs = int(1000 * Fs / np.min(fstims))
    x = np.arange(0, n, 1 / Fs)
    ys_freqs = np.sin(2 * np.pi * np.multiply.outer(fstims, x)).sum(axis=1)
    NFFT_noise = int(1000 * Fs / 11)
    np.random.seed(0)
    ys_noise = [np.random.standard_normal(n), np.random.rand(n)]
    all_kwargs = [{'sides': 'default'}, {'sides': 'onesided', 'return_line': False}, {'sides': 'twosided', 'return_line': True}]
    for ys, NFFT in [(ys_freqs, NFFT_freqs), (ys_noise, NFFT_noise)]:
        noverlap = NFFT // 2
        pad_to = int(2 ** np.ceil(np.log2(NFFT)))
        for ax, kwargs in zip(plt.figure().subplots(3), all_kwargs):
            ret = ax.psd(np.concatenate(ys), NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to, **kwargs)
            assert len(ret) == 2 + kwargs.get('return_line', False)
            ax.set(xlabel='', ylabel='')
        for ax, kwargs in zip(plt.figure().subplots(3), all_kwargs):
            ret = ax.csd(*ys, NFFT=NFFT, Fs=Fs, noverlap=noverlap, pad_to=pad_to, **kwargs)
            assert len(ret) == 2 + kwargs.get('return_line', False)
            ax.set(xlabel='', ylabel='')