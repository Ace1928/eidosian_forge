import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
def bokeh_palette_to_palette(cmap, ncolors=None, categorical=False):
    from bokeh import palettes
    categories = ['accent', 'category', 'dark', 'colorblind', 'pastel', 'set1', 'set2', 'set3', 'paired']
    cmap_categorical = any((cat in cmap.lower() for cat in categories))
    reverse = False
    if cmap.endswith('_r'):
        cmap = cmap[:-2]
        reverse = True
    inverted = not cmap_categorical and cmap.capitalize() not in palettes.mpl and (not cmap.startswith('fire'))
    if inverted:
        reverse = not reverse
    ncolors = ncolors or 256
    if cmap.startswith('tab'):
        cmap = cmap.replace('tab', 'Category')
    if cmap in palettes.all_palettes:
        palette = palettes.all_palettes[cmap]
    else:
        palette = getattr(palettes, cmap, getattr(palettes, cmap.capitalize(), None))
    if palette is None:
        raise ValueError(f'Supplied palette {cmap} not found among bokeh palettes')
    elif isinstance(palette, dict) and (cmap in palette or cmap.capitalize() in palette):
        palette = palette.get(cmap, palette.get(cmap.capitalize()))
    if isinstance(palette, dict):
        palette = palette[max(palette)]
        if not cmap_categorical:
            if len(palette) < ncolors:
                palette = polylinear_gradient(palette, ncolors)
    elif callable(palette):
        palette = palette(ncolors)
    if reverse:
        palette = palette[::-1]
    return list(resample_palette(palette, ncolors, categorical, cmap_categorical))