from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
def set_frame_props(self, props):
    """
        Set properties of the check button frames.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the check
            button frames.
        """
    _api.check_isinstance(dict, props=props)
    if 's' in props:
        props['sizes'] = np.broadcast_to(props.pop('s'), len(self.labels))
    self._frames.update(props)