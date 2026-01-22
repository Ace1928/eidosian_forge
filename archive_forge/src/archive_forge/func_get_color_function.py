import colorsys
from itertools import cycle
import numpy as np
import matplotlib as mpl
from .external import husl
from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap
def get_color_function(p0, p1):

    def color(x):
        xg = x ** gamma
        a = hue * xg * (1 - xg) / 2
        phi = 2 * np.pi * (start / 3 + rot * x)
        return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
    return color