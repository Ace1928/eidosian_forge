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
def test_axis_get_tick_params():
    axis = plt.subplot().yaxis
    initial_major_style_translated = {**axis.get_tick_params(which='major')}
    initial_minor_style_translated = {**axis.get_tick_params(which='minor')}
    translated_major_kw = axis._translate_tick_params(axis._major_tick_kw, reverse=True)
    translated_minor_kw = axis._translate_tick_params(axis._minor_tick_kw, reverse=True)
    assert translated_major_kw == initial_major_style_translated
    assert translated_minor_kw == initial_minor_style_translated
    axis.set_tick_params(labelsize=30, labelcolor='red', direction='out', which='both')
    new_major_style_translated = {**axis.get_tick_params(which='major')}
    new_minor_style_translated = {**axis.get_tick_params(which='minor')}
    new_major_style = axis._translate_tick_params(new_major_style_translated)
    new_minor_style = axis._translate_tick_params(new_minor_style_translated)
    assert initial_major_style_translated != new_major_style_translated
    assert axis._major_tick_kw == new_major_style
    assert initial_minor_style_translated != new_minor_style_translated
    assert axis._minor_tick_kw == new_minor_style