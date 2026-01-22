import itertools
import logging
import numbers
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring, colors, offsetbox
from matplotlib.artist import Artist, allow_rasterization
from matplotlib.cbook import silent_list
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import (Patch, Rectangle, Shadow, FancyBboxPatch,
from matplotlib.collections import (
from matplotlib.text import Text
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
from matplotlib.transforms import BboxTransformTo, BboxTransformFrom
from matplotlib.offsetbox import (
from matplotlib.container import ErrorbarContainer, BarContainer, StemContainer
from . import legend_handler
@staticmethod
def get_legend_handler(legend_handler_map, orig_handle):
    """
        Return a legend handler from *legend_handler_map* that
        corresponds to *orig_handler*.

        *legend_handler_map* should be a dictionary object (that is
        returned by the get_legend_handler_map method).

        It first checks if the *orig_handle* itself is a key in the
        *legend_handler_map* and return the associated value.
        Otherwise, it checks for each of the classes in its
        method-resolution-order. If no matching key is found, it
        returns ``None``.
        """
    try:
        return legend_handler_map[orig_handle]
    except (TypeError, KeyError):
        pass
    for handle_type in type(orig_handle).mro():
        try:
            return legend_handler_map[handle_type]
        except KeyError:
            pass
    return None