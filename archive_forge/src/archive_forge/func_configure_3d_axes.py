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
def configure_3d_axes(args, fig, orders):
    patch = dict(xaxis=dict(title_text=get_label(args, args['x'])), yaxis=dict(title_text=get_label(args, args['y'])), zaxis=dict(title_text=get_label(args, args['z'])))
    for letter in ['x', 'y', 'z']:
        axis = patch[letter + 'axis']
        if args['log_' + letter]:
            axis['type'] = 'log'
            if args['range_' + letter]:
                axis['range'] = [math.log(x, 10) for x in args['range_' + letter]]
        elif args['range_' + letter]:
            axis['range'] = args['range_' + letter]
        if args[letter] in orders:
            axis['categoryorder'] = 'array'
            axis['categoryarray'] = orders[args[letter]]
    fig.update_scenes(patch)