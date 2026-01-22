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
Sets a map view that contains the given geographical bounds
        with the maximum zoom level possible.

        Parameters
        ----------
        bounds: list of lists
            The lat/lon bounds in the form [[south, west], [north, east]].
        