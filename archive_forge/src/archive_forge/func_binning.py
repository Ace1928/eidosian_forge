import inspect
import re
import warnings
import matplotlib as mpl
import numpy as np
from matplotlib import (
from matplotlib.colors import Normalize, cnames
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Path, PathPatch
from matplotlib.rcsetup import validate_fontsize, validate_fonttype, validate_hatch
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from packaging.version import Version
from ...core.util import arraylike_types, cftime_types, is_number
from ...element import RGB, Polygons, Raster
from ..util import COLOR_ALIASES, RGB_HEX_REGEX
def binning(self, data, n=256):
    low = data.min() if self.vmin is None else self.vmin
    high = data.max() if self.vmax is None else self.vmax
    nbins = self._nbins
    eq_bin_edges = np.linspace(low, high, nbins + 1)
    full_hist, _ = np.histogram(data, eq_bin_edges)
    nonzero = np.nonzero(full_hist)[0]
    nhist = len(nonzero)
    if nhist > 1:
        hist = np.zeros(nhist + 1)
        hist[1:] = full_hist[nonzero]
        eq_bin_centers = np.concatenate([[0.0], (eq_bin_edges[nonzero] + eq_bin_edges[nonzero + 1]) / 2.0])
        eq_bin_centers[0] = 2 * eq_bin_centers[1] - eq_bin_centers[-1]
    else:
        hist = full_hist
        eq_bin_centers = np.convolve(eq_bin_edges, [0.5, 0.5], mode='valid')
    cdf = np.cumsum(hist)
    lo = cdf[1]
    diff = cdf[-1] - lo
    with np.errstate(divide='ignore', invalid='ignore'):
        cdf = (cdf - lo) / diff
    cdf[0] = -1.0
    lower_span = 0
    if self._rescale:
        discrete_levels = nhist
        m = -0.5 / 98.0
        c = 1.5 - 2 * m
        multiple = m * discrete_levels + c
        if multiple > 1:
            lower_span = 1 - multiple
    cdf_bins = np.linspace(lower_span, 1, n + 1)
    binning = np.interp(cdf_bins, cdf, eq_bin_centers)
    if not self._rescale:
        binning[0] = low
    binning[-1] = high
    return binning