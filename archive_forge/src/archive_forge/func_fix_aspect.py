import inspect
import re
import warnings
import matplotlib as mpl
import numpy as np
from matplotlib import (
from matplotlib.colors import Normalize, cnames
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Path, PathPatch
from matplotlib.rcsetup import validate_fontsize, validate_fonttype, validate_hatch
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from packaging.version import Version
from ...core.util import arraylike_types, cftime_types, is_number
from ...element import RGB, Polygons, Raster
from ..util import COLOR_ALIASES, RGB_HEX_REGEX
def fix_aspect(fig, nrows, ncols, title=None, extra_artists=None, vspace=0.2, hspace=0.2):
    """
    Calculate heights and widths of axes and adjust
    the size of the figure to match the aspect.
    """
    if extra_artists is None:
        extra_artists = []
    fig.canvas.draw()
    w, h = fig.get_size_inches()
    rows = resolve_rows([[ax] for ax in fig.axes])
    rs, cs = (len(rows), max([len(r) for r in rows]))
    heights = [[] for i in range(cs)]
    widths = [[] for i in range(rs)]
    for r, row in enumerate(rows):
        for c, ax in enumerate(row):
            bbox = ax.get_tightbbox(fig.canvas.get_renderer())
            heights[c].append(bbox.height)
            widths[r].append(bbox.width)
    height = max([sum(c) for c in heights]) + nrows * vspace * fig.dpi
    width = max([sum(r) for r in widths]) + ncols * hspace * fig.dpi
    aspect = height / width
    offset = 0
    if title and title.get_text():
        offset = title.get_window_extent().height / fig.dpi
    fig.set_size_inches(w, w * aspect + offset)
    fig.canvas.draw()
    if title and title.get_text():
        extra_artists = [a for a in extra_artists if a is not title]
        bbox = get_tight_bbox(fig, extra_artists)
        top = bbox.intervaly[1]
        if title and title.get_text():
            title.set_y(top / (w * aspect))