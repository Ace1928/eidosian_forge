import itertools
import io
import base64
import numpy as np
import warnings
import matplotlib
from matplotlib.colors import colorConverter
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib import ticker
def get_axes_properties(ax):
    props = {'axesbg': export_color(ax.patch.get_facecolor()), 'axesbgalpha': ax.patch.get_alpha(), 'bounds': ax.get_position().bounds, 'dynamic': ax.get_navigate(), 'axison': ax.axison, 'frame_on': ax.get_frame_on(), 'patch_visible': ax.patch.get_visible(), 'axes': [get_axis_properties(ax.xaxis), get_axis_properties(ax.yaxis)]}
    for axname in ['x', 'y']:
        axis = getattr(ax, axname + 'axis')
        domain = getattr(ax, 'get_{0}lim'.format(axname))()
        lim = domain
        if isinstance(axis.converter, matplotlib.dates.DateConverter):
            scale = 'date'
            try:
                import pandas as pd
                from pandas.tseries.converter import PeriodConverter
            except ImportError:
                pd = None
            if pd is not None and isinstance(axis.converter, PeriodConverter):
                _dates = [pd.Period(ordinal=int(d), freq=axis.freq) for d in domain]
                domain = [(d.year, d.month - 1, d.day, d.hour, d.minute, d.second, 0) for d in _dates]
            else:
                domain = [(d.year, d.month - 1, d.day, d.hour, d.minute, d.second, d.microsecond * 0.001) for d in matplotlib.dates.num2date(domain)]
        else:
            scale = axis.get_scale()
        if scale not in ['date', 'linear', 'log']:
            raise ValueError('Unknown axis scale: {0}'.format(axis.get_scale()))
        props[axname + 'scale'] = scale
        props[axname + 'lim'] = lim
        props[axname + 'domain'] = domain
    return props