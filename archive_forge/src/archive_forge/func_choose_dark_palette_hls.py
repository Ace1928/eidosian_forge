import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .miscplot import palplot
from .palettes import (color_palette, dark_palette, light_palette,
@interact
def choose_dark_palette_hls(h=(0.0, 1.0), l=(0.0, 1.0), s=(0.0, 1.0), n=(3, 17)):
    color = (h, l, s)
    if as_cmap:
        colors = dark_palette(color, 256, input='hls')
        _update_lut(cmap, colors)
        _show_cmap(cmap)
    else:
        pal[:] = dark_palette(color, n, input='hls')
        palplot(pal)