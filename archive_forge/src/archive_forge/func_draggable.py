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
def draggable(self, state=None, use_blit=False):
    """
        Set whether the annotation is draggable with the mouse.

        Parameters
        ----------
        state : bool or None
            - True or False: set the draggability.
            - None: toggle the draggability.
        use_blit : bool, default: False
            Use blitting for faster image composition. For details see
            :ref:`func-animation`.

        Returns
        -------
        DraggableAnnotation or None
            If the annotation is draggable, the corresponding
            `.DraggableAnnotation` helper is returned.
        """
    from matplotlib.offsetbox import DraggableAnnotation
    is_draggable = self._draggable is not None
    if state is None:
        state = not is_draggable
    if state:
        if self._draggable is None:
            self._draggable = DraggableAnnotation(self, use_blit)
    else:
        if self._draggable is not None:
            self._draggable.disconnect()
        self._draggable = None
    return self._draggable