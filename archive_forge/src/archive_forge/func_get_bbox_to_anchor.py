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
def get_bbox_to_anchor(self):
    """Return the bbox that the box is anchored to."""
    if self._bbox_to_anchor is None:
        return self.axes.bbox
    else:
        transform = self._bbox_to_anchor_transform
        if transform is None:
            return self._bbox_to_anchor
        else:
            return TransformedBbox(self._bbox_to_anchor, transform)