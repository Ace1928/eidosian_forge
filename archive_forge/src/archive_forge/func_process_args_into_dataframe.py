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
def process_args_into_dataframe(args, wide_mode, var_name, value_name):
    """
    After this function runs, the `all_attrables` keys of `args` all contain only
    references to columns of `df_output`. This function handles the extraction of data
    from `args["attrable"]` and column-name-generation as appropriate, and adds the
    data to `df_output` and then replaces `args["attrable"]` with the appropriate
    reference.
    """
    df_input = args['data_frame']
    df_provided = df_input is not None
    df_output = {}
    constants = {}
    ranges = []
    wide_id_vars = set()
    reserved_names = _get_reserved_col_names(args) if df_provided else set()
    if 'dimensions' in args and args['dimensions'] is None:
        if not df_provided:
            raise ValueError('No data were provided. Please provide data either with the `data_frame` or with the `dimensions` argument.')
        else:
            df_output = {col: series for col, series in df_input.items()}
    hover_data_is_dict = 'hover_data' in args and args['hover_data'] and isinstance(args['hover_data'], dict)
    if hover_data_is_dict:
        for k in args['hover_data']:
            if _isinstance_listlike(args['hover_data'][k]):
                args['hover_data'][k] = (True, args['hover_data'][k])
            if not isinstance(args['hover_data'][k], tuple):
                args['hover_data'][k] = (args['hover_data'][k], None)
            if df_provided and args['hover_data'][k][1] is not None and (k in df_input):
                raise ValueError("Ambiguous input: values for '%s' appear both in hover_data and data_frame" % k)
    for field_name in all_attrables:
        argument_list = [args.get(field_name)] if field_name not in array_attrables else args.get(field_name)
        if argument_list is None or argument_list is [None]:
            continue
        field_list = [field_name] if field_name not in array_attrables else [field_name + '_' + str(i) for i in range(len(argument_list))]
        for i, (argument, field) in enumerate(zip(argument_list, field_list)):
            length = len(df_output[next(iter(df_output))]) if len(df_output) else 0
            if argument is None:
                continue
            col_name = None
            if isinstance(argument, pd.MultiIndex):
                raise TypeError("Argument '%s' is a pandas MultiIndex. pandas MultiIndex is not supported by plotly express at the moment." % field)
            if isinstance(argument, Constant) or isinstance(argument, Range):
                col_name = _check_name_not_reserved(str(argument.label) if argument.label is not None else field, reserved_names)
                if isinstance(argument, Constant):
                    constants[col_name] = argument.value
                else:
                    ranges.append(col_name)
            elif isinstance(argument, str) or not hasattr(argument, '__len__'):
                if field_name == 'hover_data' and hover_data_is_dict and (args['hover_data'][str(argument)][1] is not None):
                    col_name = str(argument)
                    real_argument = args['hover_data'][col_name][1]
                    if length and len(real_argument) != length:
                        raise ValueError('All arguments should have the same length. The length of hover_data key `%s` is %d, whereas the length of previously-processed arguments %s is %d' % (argument, len(real_argument), str(list(df_output.keys())), length))
                    df_output[col_name] = to_unindexed_series(real_argument, col_name)
                elif not df_provided:
                    raise ValueError("String or int arguments are only possible when a DataFrame or an array is provided in the `data_frame` argument. No DataFrame was provided, but argument '%s' is of type str or int." % field)
                elif argument not in df_input.columns:
                    if wide_mode and argument in (value_name, var_name):
                        continue
                    else:
                        err_msg = "Value of '%s' is not the name of a column in 'data_frame'. Expected one of %s but received: %s" % (field, str(list(df_input.columns)), argument)
                        if argument == 'index':
                            err_msg += '\n To use the index, pass it in directly as `df.index`.'
                        raise ValueError(err_msg)
                elif length and len(df_input[argument]) != length:
                    raise ValueError('All arguments should have the same length. The length of column argument `df[%s]` is %d, whereas the length of  previously-processed arguments %s is %d' % (field, len(df_input[argument]), str(list(df_output.keys())), length))
                else:
                    col_name = str(argument)
                    df_output[col_name] = to_unindexed_series(df_input[argument], col_name)
            else:
                if df_provided and hasattr(argument, 'name'):
                    if argument is df_input.index:
                        if argument.name is None or argument.name in df_input:
                            col_name = 'index'
                        else:
                            col_name = argument.name
                        col_name = _escape_col_name(df_input, col_name, [var_name, value_name])
                    elif argument.name is not None and argument.name in df_input and (argument is df_input[argument.name]):
                        col_name = argument.name
                if col_name is None:
                    col_name = _check_name_not_reserved(field, reserved_names)
                if length and len(argument) != length:
                    raise ValueError('All arguments should have the same length. The length of argument `%s` is %d, whereas the length of  previously-processed arguments %s is %d' % (field, len(argument), str(list(df_output.keys())), length))
                df_output[str(col_name)] = to_unindexed_series(argument, str(col_name))
            assert col_name is not None, 'Data-frame processing failure, likely due to a internal bug. Please report this to https://github.com/plotly/plotly.py/issues/new and we will try to replicate and fix it.'
            if field_name not in array_attrables:
                args[field_name] = str(col_name)
            elif isinstance(args[field_name], dict):
                pass
            else:
                args[field_name][i] = str(col_name)
            if field_name != 'wide_variable':
                wide_id_vars.add(str(col_name))
    length = len(df_output[next(iter(df_output))]) if len(df_output) else 0
    df_output.update({col_name: to_unindexed_series(range(length), col_name) for col_name in ranges})
    df_output.update({col_name: to_unindexed_series([constants[col_name]] * length, col_name) for col_name in constants})
    df_output = pd.DataFrame(df_output)
    return (df_output, wide_id_vars)