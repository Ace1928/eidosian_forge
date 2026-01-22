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
def make_trace_spec(args, constructor, attrs, trace_patch):
    if constructor in [go.Scatter, go.Scatterpolar]:
        if 'render_mode' in args and (args['render_mode'] == 'webgl' or (args['render_mode'] == 'auto' and len(args['data_frame']) > 1000 and (args.get('line_shape') != 'spline') and (args['animation_frame'] is None))):
            if constructor == go.Scatter:
                constructor = go.Scattergl
                if 'orientation' in trace_patch:
                    del trace_patch['orientation']
            else:
                constructor = go.Scatterpolargl
    result = [TraceSpec(constructor, attrs, trace_patch, None)]
    for letter in ['x', 'y']:
        if 'marginal_' + letter in args and args['marginal_' + letter]:
            trace_spec = None
            axis_map = dict(xaxis='x1' if letter == 'x' else 'x2', yaxis='y1' if letter == 'y' else 'y2')
            if args['marginal_' + letter] == 'histogram':
                trace_spec = TraceSpec(constructor=go.Histogram, attrs=[letter, 'marginal_' + letter], trace_patch=dict(opacity=0.5, bingroup=letter, **axis_map), marginal=letter)
            elif args['marginal_' + letter] == 'violin':
                trace_spec = TraceSpec(constructor=go.Violin, attrs=[letter, 'hover_name', 'hover_data'], trace_patch=dict(scalegroup=letter), marginal=letter)
            elif args['marginal_' + letter] == 'box':
                trace_spec = TraceSpec(constructor=go.Box, attrs=[letter, 'hover_name', 'hover_data'], trace_patch=dict(notched=True), marginal=letter)
            elif args['marginal_' + letter] == 'rug':
                symbols = {'x': 'line-ns-open', 'y': 'line-ew-open'}
                trace_spec = TraceSpec(constructor=go.Box, attrs=[letter, 'hover_name', 'hover_data'], trace_patch=dict(fillcolor='rgba(255,255,255,0)', line={'color': 'rgba(255,255,255,0)'}, boxpoints='all', jitter=0, hoveron='points', marker={'symbol': symbols[letter]}), marginal=letter)
            if 'color' in attrs or 'color' not in args:
                if 'marker' not in trace_spec.trace_patch:
                    trace_spec.trace_patch['marker'] = dict()
                first_default_color = args['color_continuous_scale'][0]
                trace_spec.trace_patch['marker']['color'] = first_default_color
            result.append(trace_spec)
    if args.get('trendline') and args.get('trendline_scope', 'trace') == 'trace':
        result.append(make_trendline_spec(args, constructor))
    return result