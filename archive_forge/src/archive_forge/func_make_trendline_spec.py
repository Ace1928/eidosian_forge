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
def make_trendline_spec(args, constructor):
    trace_spec = TraceSpec(constructor=go.Scattergl if constructor == go.Scattergl else go.Scatter, attrs=['trendline'], trace_patch=dict(mode='lines'), marginal=None)
    if args['trendline_color_override']:
        trace_spec.trace_patch['line'] = dict(color=args['trendline_color_override'])
    return trace_spec