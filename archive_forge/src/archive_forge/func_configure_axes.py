import plotly.graph_objs as go
import plotly.io as pio
from collections import namedtuple, OrderedDict
from ._special_inputs import IdentityMap, Constant, Range
from .trendline_functions import ols, lowess, rolling, expanding, ewm
from _plotly_utils.basevalidators import ColorscaleValidator
from plotly.colors import qualitative, sequential
import math
from packaging import version
import pandas as pd
import numpy as np
from plotly._subplots import (
def configure_axes(args, constructor, fig, orders):
    configurators = {go.Scatter3d: configure_3d_axes, go.Scatterternary: configure_ternary_axes, go.Scatterpolar: configure_polar_axes, go.Scatterpolargl: configure_polar_axes, go.Barpolar: configure_polar_axes, go.Scattermapbox: configure_mapbox, go.Choroplethmapbox: configure_mapbox, go.Densitymapbox: configure_mapbox, go.Scattergeo: configure_geo, go.Choropleth: configure_geo}
    for c in cartesians:
        configurators[c] = configure_cartesian_axes
    if constructor in configurators:
        configurators[constructor](args, fig, orders)