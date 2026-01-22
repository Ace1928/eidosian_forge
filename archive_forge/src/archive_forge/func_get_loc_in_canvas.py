import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
import matplotlib.artist as martist
import matplotlib.path as mpath
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.image import BboxImage
from matplotlib.patches import (
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
def get_loc_in_canvas(self):
    offsetbox = self.offsetbox
    renderer = offsetbox.figure._get_renderer()
    bbox = offsetbox.get_bbox(renderer)
    ox, oy = offsetbox._offset
    loc_in_canvas = (ox + bbox.x0, oy + bbox.y0)
    return loc_in_canvas