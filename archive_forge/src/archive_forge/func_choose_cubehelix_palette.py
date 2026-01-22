import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .miscplot import palplot
from .palettes import (color_palette, dark_palette, light_palette,
def choose_cubehelix_palette(as_cmap=False):
    """Launch an interactive widget to create a sequential cubehelix palette.

    This corresponds with the :func:`cubehelix_palette` function. This kind
    of palette is good for data that range between relatively uninteresting
    low values and interesting high values. The cubehelix system allows the
    palette to have more hue variance across the range, which can be helpful
    for distinguishing a wider range of values.

    Requires IPython 2+ and must be used in the notebook.

    Parameters
    ----------
    as_cmap : bool
        If True, the return value is a matplotlib colormap rather than a
        list of discrete colors.

    Returns
    -------
    pal or cmap : list of colors or matplotlib colormap
        Object that can be passed to plotting functions.

    See Also
    --------
    cubehelix_palette : Create a sequential palette or colormap using the
                        cubehelix system.

    """
    pal = []
    if as_cmap:
        cmap = _init_mutable_colormap()

    @interact
    def choose_cubehelix(n_colors=IntSlider(min=2, max=16, value=9), start=FloatSlider(min=0, max=3, value=0), rot=FloatSlider(min=-1, max=1, value=0.4), gamma=FloatSlider(min=0, max=5, value=1), hue=FloatSlider(min=0, max=1, value=0.8), light=FloatSlider(min=0, max=1, value=0.85), dark=FloatSlider(min=0, max=1, value=0.15), reverse=False):
        if as_cmap:
            colors = cubehelix_palette(256, start, rot, gamma, hue, light, dark, reverse)
            _update_lut(cmap, np.c_[colors, np.ones(256)])
            _show_cmap(cmap)
        else:
            pal[:] = cubehelix_palette(n_colors, start, rot, gamma, hue, light, dark, reverse)
            palplot(pal)
    if as_cmap:
        return cmap
    return pal