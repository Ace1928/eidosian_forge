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
def set_mapbox_access_token(token):
    """
    Arguments:
        token: A Mapbox token to be used in `plotly.express.scatter_mapbox` and         `plotly.express.line_mapbox` figures. See         https://docs.mapbox.com/help/how-mapbox-works/access-tokens/ for more details
    """
    global MAPBOX_TOKEN
    MAPBOX_TOKEN = token