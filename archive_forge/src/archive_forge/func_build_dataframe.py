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
def build_dataframe(args, constructor):
    """
    Constructs a dataframe and modifies `args` in-place.

    The argument values in `args` can be either strings corresponding to
    existing columns of a dataframe, or data arrays (lists, numpy arrays,
    pandas columns, series).

    Parameters
    ----------
    args : OrderedDict
        arguments passed to the px function and subsequently modified
    constructor : graph_object trace class
        the trace type selected for this figure
    """
    for field in args:
        if field in array_attrables and args[field] is not None:
            if isinstance(args[field], dict):
                args[field] = dict(args[field])
            elif field in ['custom_data', 'hover_data'] and isinstance(args[field], str):
                args[field] = [args[field]]
            else:
                args[field] = list(args[field])
    df_provided = args['data_frame'] is not None
    needs_interchanging = False
    if df_provided and (not isinstance(args['data_frame'], pd.DataFrame)):
        if hasattr(args['data_frame'], '__dataframe__') and version.parse(pd.__version__) >= version.parse('2.0.2'):
            import pandas.api.interchange
            df_not_pandas = args['data_frame']
            args['data_frame'] = df_not_pandas.__dataframe__()
            columns = pd.Index(args['data_frame'].column_names())
            needs_interchanging = True
        elif hasattr(args['data_frame'], 'to_pandas'):
            args['data_frame'] = args['data_frame'].to_pandas()
            columns = args['data_frame'].columns
        elif hasattr(args['data_frame'], 'toPandas'):
            args['data_frame'] = args['data_frame'].toPandas()
            columns = args['data_frame'].columns
        elif hasattr(args['data_frame'], 'to_pandas_df'):
            args['data_frame'] = args['data_frame'].to_pandas_df()
            columns = args['data_frame'].columns
        else:
            args['data_frame'] = pd.DataFrame(args['data_frame'])
            columns = args['data_frame'].columns
    elif df_provided:
        columns = args['data_frame'].columns
    else:
        columns = None
    df_input = args['data_frame']
    no_x = args.get('x') is None
    no_y = args.get('y') is None
    wide_x = False if no_x else _is_col_list(columns, args['x'])
    wide_y = False if no_y else _is_col_list(columns, args['y'])
    wide_mode = False
    var_name = None
    wide_cross_name = None
    value_name = None
    hist2d_types = [go.Histogram2d, go.Histogram2dContour]
    hist1d_orientation = constructor == go.Histogram or 'ecdfmode' in args
    if constructor in cartesians:
        if wide_x and wide_y:
            raise ValueError('Cannot accept list of column references or list of columns for both `x` and `y`.')
        if df_provided and no_x and no_y:
            wide_mode = True
            if isinstance(columns, pd.MultiIndex):
                raise TypeError('Data frame columns is a pandas MultiIndex. pandas MultiIndex is not supported by plotly express at the moment.')
            args['wide_variable'] = list(columns)
            if isinstance(columns, pd.Index):
                var_name = columns.name
            else:
                var_name = None
            if var_name in [None, 'value', 'index'] or var_name in columns:
                var_name = 'variable'
            if constructor == go.Funnel:
                wide_orientation = args.get('orientation') or 'h'
            else:
                wide_orientation = args.get('orientation') or 'v'
            args['orientation'] = wide_orientation
            args['wide_cross'] = None
        elif wide_x != wide_y:
            wide_mode = True
            args['wide_variable'] = args['y'] if wide_y else args['x']
            if df_provided and args['wide_variable'] is columns:
                var_name = columns.name
            if isinstance(args['wide_variable'], pd.Index):
                args['wide_variable'] = list(args['wide_variable'])
            if var_name in [None, 'value', 'index'] or (df_provided and var_name in columns):
                var_name = 'variable'
            if hist1d_orientation:
                wide_orientation = 'v' if wide_x else 'h'
            else:
                wide_orientation = 'v' if wide_y else 'h'
            args['y' if wide_y else 'x'] = None
            args['wide_cross'] = None
            if not no_x and (not no_y):
                wide_cross_name = '__x__' if wide_y else '__y__'
    if wide_mode:
        value_name = _escape_col_name(columns, 'value', [])
        var_name = _escape_col_name(columns, var_name, [])
    if needs_interchanging:
        try:
            if wide_mode or not hasattr(args['data_frame'], 'select_columns_by_name'):
                args['data_frame'] = pd.api.interchange.from_dataframe(args['data_frame'])
            else:
                necessary_columns = {i for i in args.values() if isinstance(i, str) and i in columns}
                for field in args:
                    if args[field] is not None and field in array_attrables:
                        necessary_columns.update((i for i in args[field] if i in columns))
                columns = list(necessary_columns)
                args['data_frame'] = pd.api.interchange.from_dataframe(args['data_frame'].select_columns_by_name(columns))
        except (ImportError, NotImplementedError) as exc:
            if hasattr(df_not_pandas, 'toPandas'):
                args['data_frame'] = df_not_pandas.toPandas()
            elif hasattr(df_not_pandas, 'to_pandas_df'):
                args['data_frame'] = df_not_pandas.to_pandas_df()
            elif hasattr(df_not_pandas, 'to_pandas'):
                args['data_frame'] = df_not_pandas.to_pandas()
            else:
                raise exc
    df_input = args['data_frame']
    missing_bar_dim = None
    if constructor in [go.Scatter, go.Bar, go.Funnel] + hist2d_types and (not hist1d_orientation):
        if not wide_mode and no_x != no_y:
            for ax in ['x', 'y']:
                if args.get(ax) is None:
                    args[ax] = df_input.index if df_provided else Range()
                    if constructor == go.Bar:
                        missing_bar_dim = ax
                    elif args['orientation'] is None:
                        args['orientation'] = 'v' if ax == 'x' else 'h'
        if wide_mode and wide_cross_name is None:
            if no_x != no_y and args['orientation'] is None:
                args['orientation'] = 'v' if no_x else 'h'
            if df_provided:
                if isinstance(df_input.index, pd.MultiIndex):
                    raise TypeError('Data frame index is a pandas MultiIndex. pandas MultiIndex is not supported by plotly express at the moment.')
                args['wide_cross'] = df_input.index
            else:
                args['wide_cross'] = Range(label=_escape_col_name(df_input, 'index', [var_name, value_name]))
    no_color = False
    if type(args.get('color')) == str and args['color'] == NO_COLOR:
        no_color = True
        args['color'] = None
    df_output, wide_id_vars = process_args_into_dataframe(args, wide_mode, var_name, value_name)
    count_name = _escape_col_name(df_output, 'count', [var_name, value_name])
    if not wide_mode and missing_bar_dim and (constructor == go.Bar):
        other_dim = 'x' if missing_bar_dim == 'y' else 'y'
        if not _is_continuous(df_output, args[other_dim]):
            args[missing_bar_dim] = count_name
            df_output[count_name] = 1
        elif args['orientation'] is None:
            args['orientation'] = 'v' if missing_bar_dim == 'x' else 'h'
    if constructor in hist2d_types:
        del args['orientation']
    if wide_mode:
        wide_value_vars = [c for c in args['wide_variable'] if c not in wide_id_vars]
        del args['wide_variable']
        if wide_cross_name == '__x__':
            wide_cross_name = args['x']
        elif wide_cross_name == '__y__':
            wide_cross_name = args['y']
        else:
            wide_cross_name = args['wide_cross']
        del args['wide_cross']
        dtype = None
        for v in wide_value_vars:
            v_dtype = df_output[v].dtype.kind
            v_dtype = 'number' if v_dtype in ['i', 'f', 'u'] else v_dtype
            if dtype is None:
                dtype = v_dtype
            elif dtype != v_dtype:
                raise ValueError('Plotly Express cannot process wide-form data with columns of different type.')
        df_output = df_output.melt(id_vars=wide_id_vars, value_vars=wide_value_vars, var_name=var_name, value_name=value_name)
        assert len(df_output.columns) == len(set(df_output.columns)), 'Wide-mode name-inference failure, likely due to a internal bug. Please report this to https://github.com/plotly/plotly.py/issues/new and we will try to replicate and fix it.'
        df_output[var_name] = df_output[var_name].astype(str)
        orient_v = wide_orientation == 'v'
        if hist1d_orientation:
            args['x' if orient_v else 'y'] = value_name
            args['y' if orient_v else 'x'] = wide_cross_name
            args['color'] = args['color'] or var_name
        elif constructor in [go.Scatter, go.Funnel] + hist2d_types:
            args['x' if orient_v else 'y'] = wide_cross_name
            args['y' if orient_v else 'x'] = value_name
            if constructor != go.Histogram2d:
                args['color'] = args['color'] or var_name
            if 'line_group' in args:
                args['line_group'] = args['line_group'] or var_name
        elif constructor == go.Bar:
            if _is_continuous(df_output, value_name):
                args['x' if orient_v else 'y'] = wide_cross_name
                args['y' if orient_v else 'x'] = value_name
                args['color'] = args['color'] or var_name
            else:
                args['x' if orient_v else 'y'] = value_name
                args['y' if orient_v else 'x'] = count_name
                df_output[count_name] = 1
                args['color'] = args['color'] or var_name
        elif constructor in [go.Violin, go.Box]:
            args['x' if orient_v else 'y'] = wide_cross_name or var_name
            args['y' if orient_v else 'x'] = value_name
    if hist1d_orientation and constructor == go.Scatter:
        if args['x'] is not None and args['y'] is not None:
            args['histfunc'] = 'sum'
        elif args['x'] is None:
            args['histfunc'] = None
            args['orientation'] = 'h'
            args['x'] = count_name
            df_output[count_name] = 1
        else:
            args['histfunc'] = None
            args['orientation'] = 'v'
            args['y'] = count_name
            df_output[count_name] = 1
    if no_color:
        args['color'] = None
    args['data_frame'] = df_output
    return args