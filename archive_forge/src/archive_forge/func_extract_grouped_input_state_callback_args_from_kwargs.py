import json
from dash.development.base_component import Component
from ._validate import validate_callback
from ._grouping import flatten_grouping, make_grouping_by_index
def extract_grouped_input_state_callback_args_from_kwargs(kwargs):
    input_parameters = kwargs['inputs']
    if isinstance(input_parameters, DashDependency):
        input_parameters = [input_parameters]
    state_parameters = kwargs.get('state', None)
    if isinstance(state_parameters, DashDependency):
        state_parameters = [state_parameters]
    if isinstance(input_parameters, dict):
        if state_parameters:
            if not isinstance(state_parameters, dict):
                raise ValueError('The input argument to app.callback was a dict, but the state argument was not.\ninput and state arguments must have the same type')
            parameters = state_parameters
            parameters.update(input_parameters)
        else:
            parameters = input_parameters
        return parameters
    if isinstance(input_parameters, (list, tuple)):
        parameters = list(input_parameters)
        if state_parameters:
            if not isinstance(state_parameters, (list, tuple)):
                raise ValueError('The input argument to app.callback was a list, but the state argument was not.\ninput and state arguments must have the same type')
            parameters += list(state_parameters)
        return parameters
    raise ValueError(f'The input argument to app.callback may be a dict, list, or tuple,\nbut received value of type {type(input_parameters)}')