import colorsys
from itertools import cycle
import numpy as np
import matplotlib as mpl
from .external import husl
from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap
def light_palette(color, n_colors=6, reverse=False, as_cmap=False, input='rgb'):
    """Make a sequential palette that blends from light to ``color``.

    The ``color`` parameter can be specified in a number of ways, including
    all options for defining a color in matplotlib and several additional
    color spaces that are handled by seaborn. You can also use the database
    of named colors from the XKCD color survey.

    If you are using a Jupyter notebook, you can also choose this palette
    interactively with the :func:`choose_light_palette` function.

    Parameters
    ----------
    color : base color for high values
        hex code, html color name, or tuple in `input` space.
    n_colors : int, optional
        number of colors in the palette
    reverse : bool, optional
        if True, reverse the direction of the blend
    as_cmap : bool, optional
        If True, return a :class:`matplotlib.colors.ListedColormap`.
    input : {'rgb', 'hls', 'husl', xkcd'}
        Color space to interpret the input color. The first three options
        apply to tuple inputs and the latter applies to string inputs.

    Returns
    -------
    palette
        list of RGB tuples or :class:`matplotlib.colors.ListedColormap`

    See Also
    --------
    dark_palette : Create a sequential palette with dark low values.
    diverging_palette : Create a diverging palette with two colors.

    Examples
    --------
    .. include:: ../docstrings/light_palette.rst

    """
    rgb = _color_to_rgb(color, input)
    hue, sat, _ = husl.rgb_to_husl(*rgb)
    gray_s, gray_l = (0.15 * sat, 95)
    gray = _color_to_rgb((hue, gray_s, gray_l), input='husl')
    colors = [rgb, gray] if reverse else [gray, rgb]
    return blend_palette(colors, n_colors, as_cmap)