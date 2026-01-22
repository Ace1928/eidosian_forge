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
def configure_cartesian_axes(args, fig, orders):
    if 'marginal_x' in args and args['marginal_x'] or ('marginal_y' in args and args['marginal_y']):
        configure_cartesian_marginal_axes(args, fig, orders)
        return
    y_title = get_decorated_label(args, args['y'], 'y')
    for yaxis in fig.select_yaxes(col=1):
        yaxis.update(title_text=y_title)
        set_cartesian_axis_opts(args, yaxis, 'y', orders)
    x_title = get_decorated_label(args, args['x'], 'x')
    for xaxis in fig.select_xaxes(row=1):
        if 'is_timeline' not in args:
            xaxis.update(title_text=x_title)
        set_cartesian_axis_opts(args, xaxis, 'x', orders)
    if 'log_x' in args and args['log_x']:
        fig.update_xaxes(type='log')
    if 'log_y' in args and args['log_y']:
        fig.update_yaxes(type='log')
    if 'is_timeline' in args:
        fig.update_xaxes(type='date')
    if 'ecdfmode' in args:
        if args['orientation'] == 'v':
            fig.update_yaxes(rangemode='tozero')
        else:
            fig.update_xaxes(rangemode='tozero')