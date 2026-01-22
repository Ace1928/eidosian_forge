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
def configure_polar_axes(args, fig, orders):
    patch = dict(angularaxis=dict(direction=args['direction'], rotation=args['start_angle']), radialaxis=dict())
    for var, axis in [('r', 'radialaxis'), ('theta', 'angularaxis')]:
        if args[var] in orders:
            patch[axis]['categoryorder'] = 'array'
            patch[axis]['categoryarray'] = orders[args[var]]
    radialaxis = patch['radialaxis']
    if args['log_r']:
        radialaxis['type'] = 'log'
        if args['range_r']:
            radialaxis['range'] = [math.log(x, 10) for x in args['range_r']]
    elif args['range_r']:
        radialaxis['range'] = args['range_r']
    if args['range_theta']:
        patch['sector'] = args['range_theta']
    fig.update_polars(patch)