import colorsys
from itertools import cycle
import numpy as np
import matplotlib as mpl
from .external import husl
from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap
def crayon_palette(colors):
    """Make a palette with color names from Crayola crayons.

    Colors are taken from here:
    https://en.wikipedia.org/wiki/List_of_Crayola_crayon_colors

    This is just a simple wrapper around the `seaborn.crayons` dictionary.

    Parameters
    ----------
    colors : list of strings
        List of keys in the `seaborn.crayons` dictionary.

    Returns
    -------
    palette
        A list of colors as RGB tuples.

    See Also
    --------
    xkcd_palette : Make a palette with named colors from the XKCD color survey.

    """
    palette = [crayons[name] for name in colors]
    return color_palette(palette, len(palette))