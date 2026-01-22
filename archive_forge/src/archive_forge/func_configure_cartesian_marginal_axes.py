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
def configure_cartesian_marginal_axes(args, fig, orders):
    if 'histogram' in [args['marginal_x'], args['marginal_y']]:
        fig.layout['barmode'] = 'overlay'
    nrows = len(fig._grid_ref)
    ncols = len(fig._grid_ref[0])
    for yaxis in fig.select_yaxes(col=1):
        set_cartesian_axis_opts(args, yaxis, 'y', orders)
    for xaxis in fig.select_xaxes(row=1):
        set_cartesian_axis_opts(args, xaxis, 'x', orders)
    if args['marginal_x']:
        fig.update_yaxes(showticklabels=False, showline=False, ticks='', range=None, row=nrows)
        if args['template'].layout.yaxis.showgrid is None:
            fig.update_yaxes(showgrid=args['marginal_x'] == 'histogram', row=nrows)
        if args['template'].layout.xaxis.showgrid is None:
            fig.update_xaxes(showgrid=True, row=nrows)
    if args['marginal_y']:
        fig.update_xaxes(showticklabels=False, showline=False, ticks='', range=None, col=ncols)
        if args['template'].layout.xaxis.showgrid is None:
            fig.update_xaxes(showgrid=args['marginal_y'] == 'histogram', col=ncols)
        if args['template'].layout.yaxis.showgrid is None:
            fig.update_yaxes(showgrid=True, col=ncols)
    y_title = get_decorated_label(args, args['y'], 'y')
    if args['marginal_x']:
        fig.update_yaxes(title_text=y_title, row=1, col=1)
    else:
        for row in range(1, nrows + 1):
            fig.update_yaxes(title_text=y_title, row=row, col=1)
    x_title = get_decorated_label(args, args['x'], 'x')
    if args['marginal_y']:
        fig.update_xaxes(title_text=x_title, row=1, col=1)
    else:
        for col in range(1, ncols + 1):
            fig.update_xaxes(title_text=x_title, row=1, col=col)
    if 'log_x' in args and args['log_x']:
        fig.update_xaxes(type='log')
    if 'log_y' in args and args['log_y']:
        fig.update_yaxes(type='log')
    matches_y = 'y' + str(ncols + 1)
    if args['marginal_x']:
        for row in range(2, nrows + 1, 2):
            fig.update_yaxes(matches=matches_y, type=None, row=row)
    if args['marginal_y']:
        for col in range(2, ncols + 1, 2):
            fig.update_xaxes(matches='x2', type=None, col=col)