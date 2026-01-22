import copy
import asyncio
import json
import xyzservices
from datetime import date, timedelta
from math import isnan
from branca.colormap import linear, ColorMap
from IPython.display import display
import warnings
from ipywidgets import (
from ipywidgets.widgets.trait_types import InstanceDict
from ipywidgets.embed import embed_minimal_html
from traitlets import (
from ._version import EXTENSION_VERSION
from .projections import projections
def remove_control(self, control):
    """Remove a control from the map.

        .. deprecated :: 0.17.0
           Use remove method instead.

        Parameters
        ----------
        control: Control instance
            The control to remove.
        """
    warnings.warn('remove_control is deprecated, use remove instead', DeprecationWarning)
    self.remove(control)