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
def colorcet_cmap_to_palette(cmap, ncolors=None, categorical=False):
    from colorcet import palette
    categories = ['glasbey']
    ncolors = ncolors or 256
    cmap_categorical = any((c in cmap for c in categories))
    if cmap.endswith('_r'):
        palette = list(reversed(palette[cmap[:-2]]))
    else:
        palette = palette[cmap]
    return resample_palette(palette, ncolors, categorical, cmap_categorical)