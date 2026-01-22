import importlib
import warnings
from typing import Any, Dict
import matplotlib as mpl
import numpy as np
import packaging
from matplotlib.colors import to_hex
from scipy.stats import mode, rankdata
from scipy.interpolate import CubicSpline
from ..rcparams import rcParams
from ..stats.density_utils import kde
from ..stats import hdi
def set_bokeh_circular_ticks_labels(ax, hist, labels):
    """Place ticks and ticklabels on Bokeh's circular histogram."""
    ticks = np.linspace(-np.pi, np.pi, len(labels), endpoint=False)
    ax.annular_wedge(x=0, y=0, inner_radius=0, outer_radius=np.max(hist) * 1.1, start_angle=ticks, end_angle=ticks, line_color='grey')
    radii_circles = np.linspace(0, np.max(hist) * 1.1, 4)
    ax.circle(0, 0, radius=radii_circles, fill_color=None, line_color='grey')
    offset = np.max(hist * 1.05) * 0.15
    ticks_labels_pos_1 = np.max(hist * 1.05)
    ticks_labels_pos_2 = ticks_labels_pos_1 * np.sqrt(2) / 2
    ax.text([ticks_labels_pos_1 + offset, ticks_labels_pos_2 + offset, 0, -ticks_labels_pos_2 - offset, -ticks_labels_pos_1 - offset, -ticks_labels_pos_2 - offset, 0, ticks_labels_pos_2 + offset], [0, ticks_labels_pos_2 + offset / 2, ticks_labels_pos_1 + offset, ticks_labels_pos_2 + offset / 2, 0, -ticks_labels_pos_2 - offset, -ticks_labels_pos_1 - offset, -ticks_labels_pos_2 - offset], text=labels, text_align='center')
    return ax