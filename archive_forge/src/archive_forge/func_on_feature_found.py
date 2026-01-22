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
def on_feature_found(self, callback, remove=False):
    """Add a found feature event listener for searching in GeoJSON layer.

        Parameters
        ----------
        callback : callable
            Callback function that will be called on found event when searching in GeoJSON layer.
        remove: boolean
            Whether to remove this callback or not. Defaults to False.
        """
    self._location_found_callbacks.register_callback(callback, remove=remove)