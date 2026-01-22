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
def get_path_style(path, fill=True):
    """Get the style dictionary for matplotlib path objects"""
    style = {}
    style['alpha'] = path.get_alpha()
    if style['alpha'] is None:
        style['alpha'] = 1
    style['edgecolor'] = export_color(path.get_edgecolor())
    if fill:
        style['facecolor'] = export_color(path.get_facecolor())
    else:
        style['facecolor'] = 'none'
    style['edgewidth'] = path.get_linewidth()
    style['dasharray'] = get_dasharray(path)
    style['zorder'] = path.get_zorder()
    return style