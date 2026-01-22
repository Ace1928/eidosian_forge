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
def set_bbox_to_anchor(self, bbox, transform=None):
    """
        Set the bbox that the box is anchored to.

        *bbox* can be a Bbox instance, a list of [left, bottom, width,
        height], or a list of [left, bottom] where the width and
        height will be assumed to be zero. The bbox will be
        transformed to display coordinate by the given transform.
        """
    if bbox is None or isinstance(bbox, BboxBase):
        self._bbox_to_anchor = bbox
    else:
        try:
            l = len(bbox)
        except TypeError as err:
            raise ValueError(f'Invalid bbox: {bbox}') from err
        if l == 2:
            bbox = [bbox[0], bbox[1], 0, 0]
        self._bbox_to_anchor = Bbox.from_bounds(*bbox)
    self._bbox_to_anchor_transform = transform
    self.stale = True