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
def make_mapping(args, variable):
    if variable == 'line_group' or variable == 'animation_frame':
        return Mapping(show_in_trace_name=False, grouper=args[variable], val_map={}, sequence=[''], variable=variable, updater=lambda trace, v: v, facet=None)
    if variable == 'facet_row' or variable == 'facet_col':
        letter = 'x' if variable == 'facet_col' else 'y'
        return Mapping(show_in_trace_name=False, variable=letter, grouper=args[variable], val_map={}, sequence=[i for i in range(1, 1000)], updater=lambda trace, v: v, facet='row' if variable == 'facet_row' else 'col')
    parent, variable, *other_variables = variable.split('.')
    vprefix = variable
    arg_name = variable
    if variable == 'color':
        vprefix = 'color_discrete'
    if variable == 'dash':
        arg_name = 'line_dash'
        vprefix = 'line_dash'
    if variable in ['pattern', 'shape']:
        arg_name = 'pattern_shape'
        vprefix = 'pattern_shape'
    if args[vprefix + '_map'] == 'identity':
        val_map = IdentityMap()
    else:
        val_map = args[vprefix + '_map'].copy()
    return Mapping(show_in_trace_name=True, variable=variable, grouper=args[arg_name], val_map=val_map, sequence=args[vprefix + '_sequence'], updater=lambda trace, v: trace.update({parent: {'.'.join([variable] + other_variables): v}}), facet=None)