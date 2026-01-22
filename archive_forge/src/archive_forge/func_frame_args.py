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
def frame_args(duration):
    return {'frame': {'duration': duration, 'redraw': constructor != go.Scatter}, 'mode': 'immediate', 'fromcurrent': True, 'transition': {'duration': duration, 'easing': 'linear'}}