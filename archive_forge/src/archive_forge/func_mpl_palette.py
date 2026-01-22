import colorsys
from itertools import cycle
import numpy as np
import matplotlib as mpl
from .external import husl
from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap
def mpl_palette(name, n_colors=6, as_cmap=False):
    """
    Return a palette or colormap from the matplotlib registry.

    For continuous palettes, evenly-spaced discrete samples are chosen while
    excluding the minimum and maximum value in the colormap to provide better
    contrast at the extremes.

    For qualitative palettes (e.g. those from colorbrewer), exact values are
    indexed (rather than interpolated), but fewer than `n_colors` can be returned
    if the palette does not define that many.

    Parameters
    ----------
    name : string
        Name of the palette. This should be a named matplotlib colormap.
    n_colors : int
        Number of discrete colors in the palette.

    Returns
    -------
    list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    Examples
    --------
    .. include:: ../docstrings/mpl_palette.rst

    """
    if name.endswith('_d'):
        sub_name = name[:-2]
        if sub_name.endswith('_r'):
            reverse = True
            sub_name = sub_name[:-2]
        else:
            reverse = False
        pal = color_palette(sub_name, 2) + ['#333333']
        if reverse:
            pal = pal[::-1]
        cmap = blend_palette(pal, n_colors, as_cmap=True)
    else:
        cmap = get_colormap(name)
    if name in MPL_QUAL_PALS:
        bins = np.linspace(0, 1, MPL_QUAL_PALS[name])[:n_colors]
    else:
        bins = np.linspace(0, 1, int(n_colors) + 2)[1:-1]
    palette = list(map(tuple, cmap(bins)[:, :3]))
    if as_cmap:
        return cmap
    else:
        return _ColorPalette(palette)