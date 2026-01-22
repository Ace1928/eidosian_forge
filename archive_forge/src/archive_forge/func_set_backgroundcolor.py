import functools
import logging
import math
from numbers import Real
import weakref
import numpy as np
import matplotlib as mpl
from . import _api, artist, cbook, _docstring
from .artist import Artist
from .font_manager import FontProperties
from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from .textpath import TextPath, TextToPath  # noqa # Logically located here
from .transforms import (
def set_backgroundcolor(self, color):
    """
        Set the background color of the text by updating the bbox.

        Parameters
        ----------
        color : color

        See Also
        --------
        .set_bbox : To change the position of the bounding box
        """
    if self._bbox_patch is None:
        self.set_bbox(dict(facecolor=color, edgecolor=color))
    else:
        self._bbox_patch.update(dict(facecolor=color))
    self._update_clip_properties()
    self.stale = True