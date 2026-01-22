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
def set_bbox(self, rectprops):
    """
        Draw a bounding box around self.

        Parameters
        ----------
        rectprops : dict with properties for `.patches.FancyBboxPatch`
             The default boxstyle is 'square'. The mutation
             scale of the `.patches.FancyBboxPatch` is set to the fontsize.

        Examples
        --------
        ::

            t.set_bbox(dict(facecolor='red', alpha=0.5))
        """
    if rectprops is not None:
        props = rectprops.copy()
        boxstyle = props.pop('boxstyle', None)
        pad = props.pop('pad', None)
        if boxstyle is None:
            boxstyle = 'square'
            if pad is None:
                pad = 4
            pad /= self.get_size()
        elif pad is None:
            pad = 0.3
        if isinstance(boxstyle, str) and 'pad' not in boxstyle:
            boxstyle += ',pad=%0.2f' % pad
        self._bbox_patch = FancyBboxPatch((0, 0), 1, 1, boxstyle=boxstyle, transform=IdentityTransform(), **props)
    else:
        self._bbox_patch = None
    self._update_clip_properties()