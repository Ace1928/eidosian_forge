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
def make_trace_kwargs(args, trace_spec, trace_data, mapping_labels, sizeref):
    """Populates a dict with arguments to update trace

    Parameters
    ----------
    args : dict
        args to be used for the trace
    trace_spec : NamedTuple
        which kind of trace to be used (has constructor, marginal etc.
        attributes)
    trace_data : pandas DataFrame
        data
    mapping_labels : dict
        to be used for hovertemplate
    sizeref : float
        marker sizeref

    Returns
    -------
    trace_patch : dict
        dict to be used to update trace
    fit_results : dict
        fit information to be used for trendlines
    """
    if 'line_close' in args and args['line_close']:
        trace_data = pd.concat([trace_data, trace_data.iloc[:1]])
    trace_patch = trace_spec.trace_patch.copy() or {}
    fit_results = None
    hover_header = ''
    for attr_name in trace_spec.attrs:
        attr_value = args[attr_name]
        attr_label = get_decorated_label(args, attr_value, attr_name)
        if attr_name == 'dimensions':
            dims = [(name, column) for name, column in trace_data.items() if (not attr_value or name in attr_value) and (trace_spec.constructor != go.Parcoords or _is_continuous(args['data_frame'], name)) and (trace_spec.constructor != go.Parcats or (attr_value is not None and name in attr_value) or len(args['data_frame'][name].unique()) <= args['dimensions_max_cardinality'])]
            trace_patch['dimensions'] = [dict(label=get_label(args, name), values=column) for name, column in dims]
            if trace_spec.constructor == go.Splom:
                for d in trace_patch['dimensions']:
                    d['axis'] = dict(matches=True)
                mapping_labels['%{xaxis.title.text}'] = '%{x}'
                mapping_labels['%{yaxis.title.text}'] = '%{y}'
        elif attr_value is not None:
            if attr_name == 'size':
                if 'marker' not in trace_patch:
                    trace_patch['marker'] = dict()
                trace_patch['marker']['size'] = trace_data[attr_value]
                trace_patch['marker']['sizemode'] = 'area'
                trace_patch['marker']['sizeref'] = sizeref
                mapping_labels[attr_label] = '%{marker.size}'
            elif attr_name == 'marginal_x':
                if trace_spec.constructor == go.Histogram:
                    mapping_labels['count'] = '%{y}'
            elif attr_name == 'marginal_y':
                if trace_spec.constructor == go.Histogram:
                    mapping_labels['count'] = '%{x}'
            elif attr_name == 'trendline':
                if args['x'] and args['y'] and (len(trace_data[[args['x'], args['y']]].dropna()) > 1):
                    sorted_trace_data = trace_data.sort_values(by=args['x'])
                    y = sorted_trace_data[args['y']].values
                    x = sorted_trace_data[args['x']].values
                    if x.dtype.type == np.datetime64:
                        x = x.astype(np.int64) / 10 ** 9
                    elif x.dtype.type == np.object_:
                        try:
                            x = x.astype(np.float64)
                        except ValueError:
                            raise ValueError("Could not convert value of 'x' ('%s') into a numeric type. If 'x' contains stringified dates, please convert to a datetime column." % args['x'])
                    if y.dtype.type == np.object_:
                        try:
                            y = y.astype(np.float64)
                        except ValueError:
                            raise ValueError("Could not convert value of 'y' into a numeric type.")
                    non_missing = np.logical_not(np.logical_or(np.isnan(y), np.isnan(x)))
                    trace_patch['x'] = sorted_trace_data[args['x']][non_missing]
                    trendline_function = trendline_functions[attr_value]
                    y_out, hover_header, fit_results = trendline_function(args['trendline_options'], sorted_trace_data[args['x']], x, y, args['x'], args['y'], non_missing)
                    assert len(y_out) == len(trace_patch['x']), 'missing-data-handling failure in trendline code'
                    trace_patch['y'] = y_out
                    mapping_labels[get_label(args, args['x'])] = '%{x}'
                    mapping_labels[get_label(args, args['y'])] = '%{y} <b>(trend)</b>'
            elif attr_name.startswith('error'):
                error_xy = attr_name[:7]
                arr = 'arrayminus' if attr_name.endswith('minus') else 'array'
                if error_xy not in trace_patch:
                    trace_patch[error_xy] = {}
                trace_patch[error_xy][arr] = trace_data[attr_value]
            elif attr_name == 'custom_data':
                if len(attr_value) > 0:
                    trace_patch['customdata'] = trace_data[attr_value]
            elif attr_name == 'hover_name':
                if trace_spec.constructor not in [go.Histogram, go.Histogram2d, go.Histogram2dContour]:
                    trace_patch['hovertext'] = trace_data[attr_value]
                    if hover_header == '':
                        hover_header = '<b>%{hovertext}</b><br><br>'
            elif attr_name == 'hover_data':
                if trace_spec.constructor not in [go.Histogram, go.Histogram2d, go.Histogram2dContour]:
                    hover_is_dict = isinstance(attr_value, dict)
                    customdata_cols = args.get('custom_data') or []
                    for col in attr_value:
                        if hover_is_dict and (not attr_value[col]):
                            continue
                        if col in [args.get('x'), args.get('y'), args.get('z'), args.get('base')]:
                            continue
                        try:
                            position = args['custom_data'].index(col)
                        except (ValueError, AttributeError, KeyError):
                            position = len(customdata_cols)
                            customdata_cols.append(col)
                        attr_label_col = get_decorated_label(args, col, None)
                        mapping_labels[attr_label_col] = '%%{customdata[%d]}' % position
                    if len(customdata_cols) > 0:
                        trace_patch['customdata'] = trace_data[customdata_cols]
            elif attr_name == 'color':
                if trace_spec.constructor in [go.Choropleth, go.Choroplethmapbox]:
                    trace_patch['z'] = trace_data[attr_value]
                    trace_patch['coloraxis'] = 'coloraxis1'
                    mapping_labels[attr_label] = '%{z}'
                elif trace_spec.constructor in [go.Sunburst, go.Treemap, go.Icicle, go.Pie, go.Funnelarea]:
                    if 'marker' not in trace_patch:
                        trace_patch['marker'] = dict()
                    if args.get('color_is_continuous'):
                        trace_patch['marker']['colors'] = trace_data[attr_value]
                        trace_patch['marker']['coloraxis'] = 'coloraxis1'
                        mapping_labels[attr_label] = '%{color}'
                    else:
                        trace_patch['marker']['colors'] = []
                        if args['color_discrete_map'] is not None:
                            mapping = args['color_discrete_map'].copy()
                        else:
                            mapping = {}
                        for cat in trace_data[attr_value]:
                            if mapping.get(cat) is None:
                                mapping[cat] = args['color_discrete_sequence'][len(mapping) % len(args['color_discrete_sequence'])]
                            trace_patch['marker']['colors'].append(mapping[cat])
                else:
                    colorable = 'marker'
                    if trace_spec.constructor in [go.Parcats, go.Parcoords]:
                        colorable = 'line'
                    if colorable not in trace_patch:
                        trace_patch[colorable] = dict()
                    trace_patch[colorable]['color'] = trace_data[attr_value]
                    trace_patch[colorable]['coloraxis'] = 'coloraxis1'
                    mapping_labels[attr_label] = '%%{%s.color}' % colorable
            elif attr_name == 'animation_group':
                trace_patch['ids'] = trace_data[attr_value]
            elif attr_name == 'locations':
                trace_patch[attr_name] = trace_data[attr_value]
                mapping_labels[attr_label] = '%{location}'
            elif attr_name == 'values':
                trace_patch[attr_name] = trace_data[attr_value]
                _label = 'value' if attr_label == 'values' else attr_label
                mapping_labels[_label] = '%{value}'
            elif attr_name == 'parents':
                trace_patch[attr_name] = trace_data[attr_value]
                _label = 'parent' if attr_label == 'parents' else attr_label
                mapping_labels[_label] = '%{parent}'
            elif attr_name == 'ids':
                trace_patch[attr_name] = trace_data[attr_value]
                _label = 'id' if attr_label == 'ids' else attr_label
                mapping_labels[_label] = '%{id}'
            elif attr_name == 'names':
                if trace_spec.constructor in [go.Sunburst, go.Treemap, go.Icicle, go.Pie, go.Funnelarea]:
                    trace_patch['labels'] = trace_data[attr_value]
                    _label = 'label' if attr_label == 'names' else attr_label
                    mapping_labels[_label] = '%{label}'
                else:
                    trace_patch[attr_name] = trace_data[attr_value]
            else:
                trace_patch[attr_name] = trace_data[attr_value]
                mapping_labels[attr_label] = '%%{%s}' % attr_name
        elif trace_spec.constructor == go.Histogram and attr_name in ['x', 'y'] or (trace_spec.constructor in [go.Histogram2d, go.Histogram2dContour] and attr_name == 'z'):
            mapping_labels[attr_label] = '%%{%s}' % attr_name
    if trace_spec.constructor not in [go.Parcoords, go.Parcats]:
        mapping_labels_copy = OrderedDict(mapping_labels)
        if args['hover_data'] and isinstance(args['hover_data'], dict):
            for k, v in mapping_labels.items():
                k_args = invert_label(args, k)
                if k_args in args['hover_data']:
                    formatter = args['hover_data'][k_args][0]
                    if formatter:
                        if isinstance(formatter, str):
                            mapping_labels_copy[k] = v.replace('}', '%s}' % formatter)
                    else:
                        _ = mapping_labels_copy.pop(k)
        hover_lines = [k + '=' + v for k, v in mapping_labels_copy.items()]
        trace_patch['hovertemplate'] = hover_header + '<br>'.join(hover_lines)
        trace_patch['hovertemplate'] += '<extra></extra>'
    return (trace_patch, fit_results)