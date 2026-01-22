import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .miscplot import palplot
from .palettes import (color_palette, dark_palette, light_palette,
def choose_light_palette(input='husl', as_cmap=False):
    """Launch an interactive widget to create a light sequential palette.

    This corresponds with the :func:`light_palette` function. This kind
    of palette is good for data that range between relatively uninteresting
    low values and interesting high values.

    Requires IPython 2+ and must be used in the notebook.

    Parameters
    ----------
    input : {'husl', 'hls', 'rgb'}
        Color space for defining the seed value. Note that the default is
        different than the default input for :func:`light_palette`.
    as_cmap : bool
        If True, the return value is a matplotlib colormap rather than a
        list of discrete colors.

    Returns
    -------
    pal or cmap : list of colors or matplotlib colormap
        Object that can be passed to plotting functions.

    See Also
    --------
    light_palette : Create a sequential palette with bright low values.
    dark_palette : Create a sequential palette with dark low values.
    cubehelix_palette : Create a sequential palette or colormap using the
                        cubehelix system.

    """
    pal = []
    if as_cmap:
        cmap = _init_mutable_colormap()
    if input == 'rgb':

        @interact
        def choose_light_palette_rgb(r=(0.0, 1.0), g=(0.0, 1.0), b=(0.0, 1.0), n=(3, 17)):
            color = (r, g, b)
            if as_cmap:
                colors = light_palette(color, 256, input='rgb')
                _update_lut(cmap, colors)
                _show_cmap(cmap)
            else:
                pal[:] = light_palette(color, n, input='rgb')
                palplot(pal)
    elif input == 'hls':

        @interact
        def choose_light_palette_hls(h=(0.0, 1.0), l=(0.0, 1.0), s=(0.0, 1.0), n=(3, 17)):
            color = (h, l, s)
            if as_cmap:
                colors = light_palette(color, 256, input='hls')
                _update_lut(cmap, colors)
                _show_cmap(cmap)
            else:
                pal[:] = light_palette(color, n, input='hls')
                palplot(pal)
    elif input == 'husl':

        @interact
        def choose_light_palette_husl(h=(0, 359), s=(0, 99), l=(0, 99), n=(3, 17)):
            color = (h, s, l)
            if as_cmap:
                colors = light_palette(color, 256, input='husl')
                _update_lut(cmap, colors)
                _show_cmap(cmap)
            else:
                pal[:] = light_palette(color, n, input='husl')
                palplot(pal)
    if as_cmap:
        return cmap
    return pal