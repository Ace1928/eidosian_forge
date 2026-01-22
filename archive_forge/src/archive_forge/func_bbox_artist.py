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
@_api.deprecated('3.7', alternative='patches.bbox_artist')
def bbox_artist(*args, **kwargs):
    if DEBUG:
        mbbox_artist(*args, **kwargs)