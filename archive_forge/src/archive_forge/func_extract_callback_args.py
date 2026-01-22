import json
from dash.development.base_component import Component
from ._validate import validate_callback
from ._grouping import flatten_grouping, make_grouping_by_index
def extract_callback_args(args, kwargs, name, type_):
    """Extract arguments for callback from a name and type"""
    parameters = kwargs.get(name, [])
    if parameters:
        if not isinstance(parameters, (list, tuple)):
            return [parameters]
    else:
        while args and isinstance(args[0], type_):
            parameters.append(args.pop(0))
    return parameters