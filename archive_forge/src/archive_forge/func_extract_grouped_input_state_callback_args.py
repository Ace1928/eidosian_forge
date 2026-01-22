import json
from dash.development.base_component import Component
from ._validate import validate_callback
from ._grouping import flatten_grouping, make_grouping_by_index
def extract_grouped_input_state_callback_args(args, kwargs):
    if 'inputs' in kwargs:
        return extract_grouped_input_state_callback_args_from_kwargs(kwargs)
    if 'state' in kwargs:
        raise ValueError('The state keyword argument may not be provided without the input keyword argument')
    return extract_grouped_input_state_callback_args_from_args(args)