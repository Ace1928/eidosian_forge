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
def configure_mapbox(args, fig, orders):
    center = args['center']
    if not center and 'lat' in args and ('lon' in args):
        center = dict(lat=args['data_frame'][args['lat']].mean(), lon=args['data_frame'][args['lon']].mean())
    fig.update_mapboxes(accesstoken=MAPBOX_TOKEN, center=center, zoom=args['zoom'], style=args['mapbox_style'])