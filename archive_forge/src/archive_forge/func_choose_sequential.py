import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .miscplot import palplot
from .palettes import (color_palette, dark_palette, light_palette,
@interact
def choose_sequential(name=opts, n=(2, 18), desat=FloatSlider(min=0, max=1, value=1), variant=variants):
    if variant == 'reverse':
        name += '_r'
    elif variant == 'dark':
        name += '_d'
    if as_cmap:
        colors = color_palette(name, 256, desat)
        _update_lut(cmap, np.c_[colors, np.ones(256)])
        _show_cmap(cmap)
    else:
        pal[:] = color_palette(name, n, desat)
        palplot(pal)