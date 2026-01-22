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
def on_dblclick(self, callback, remove=False):
    """Add a double-click event listener.

        Parameters
        ----------
        callback : callable
            Callback function that will be called on double-click event.
        remove: boolean
            Whether to remove this callback or not. Defaults to False.
        """
    self._dblclick_callbacks.register_callback(callback, remove=remove)