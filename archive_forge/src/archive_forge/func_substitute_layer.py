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
def substitute_layer(self, old, new):
    """Replace a layer with another one on the map.

        .. deprecated :: 0.17.0
           Use substitute method instead.

        Parameters
        ----------
        old: Layer instance
            The old layer to remove.
        new: Layer instance
            The new layer to add.
        """
    warnings.warn('substitute_layer is deprecated, use substitute instead', DeprecationWarning)
    self.substitute(old, new)