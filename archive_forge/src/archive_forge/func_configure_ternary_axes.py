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
def configure_ternary_axes(args, fig, orders):
    fig.update_ternaries(aaxis=dict(title_text=get_label(args, args['a'])), baxis=dict(title_text=get_label(args, args['b'])), caxis=dict(title_text=get_label(args, args['c'])))