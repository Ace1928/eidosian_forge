import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
def from_levels_and_colors(levels, colors, extend='neither'):
    """
    A helper routine to generate a cmap and a norm instance which
    behave similar to contourf's levels and colors arguments.

    Parameters
    ----------
    levels : sequence of numbers
        The quantization levels used to construct the `BoundaryNorm`.
        Value ``v`` is quantized to level ``i`` if ``lev[i] <= v < lev[i+1]``.
    colors : sequence of colors
        The fill color to use for each level. If *extend* is "neither" there
        must be ``n_level - 1`` colors. For an *extend* of "min" or "max" add
        one extra color, and for an *extend* of "both" add two colors.
    extend : {'neither', 'min', 'max', 'both'}, optional
        The behaviour when a value falls out of range of the given levels.
        See `~.Axes.contourf` for details.

    Returns
    -------
    cmap : `~matplotlib.colors.Colormap`
    norm : `~matplotlib.colors.Normalize`
    """
    slice_map = {'both': slice(1, -1), 'min': slice(1, None), 'max': slice(0, -1), 'neither': slice(0, None)}
    _api.check_in_list(slice_map, extend=extend)
    color_slice = slice_map[extend]
    n_data_colors = len(levels) - 1
    n_expected = n_data_colors + color_slice.start - (color_slice.stop or 0)
    if len(colors) != n_expected:
        raise ValueError(f'With extend == {extend!r} and {len(levels)} levels, expected {n_expected} colors, but got {len(colors)}')
    cmap = ListedColormap(colors[color_slice], N=n_data_colors)
    if extend in ['min', 'both']:
        cmap.set_under(colors[0])
    else:
        cmap.set_under('none')
    if extend in ['max', 'both']:
        cmap.set_over(colors[-1])
    else:
        cmap.set_over('none')
    cmap.colorbar_extend = extend
    norm = BoundaryNorm(levels, ncolors=n_data_colors)
    return (cmap, norm)