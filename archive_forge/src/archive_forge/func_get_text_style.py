import itertools
import io
import base64
import numpy as np
import warnings
import matplotlib
from matplotlib.colors import colorConverter
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib import ticker
def get_text_style(text):
    """Return the text style dict for a text instance"""
    style = {}
    style['alpha'] = text.get_alpha()
    if style['alpha'] is None:
        style['alpha'] = 1
    style['fontsize'] = text.get_size()
    style['color'] = export_color(text.get_color())
    style['halign'] = text.get_horizontalalignment()
    style['valign'] = text.get_verticalalignment()
    style['malign'] = text._multialignment
    style['rotation'] = text.get_rotation()
    style['zorder'] = text.get_zorder()
    return style