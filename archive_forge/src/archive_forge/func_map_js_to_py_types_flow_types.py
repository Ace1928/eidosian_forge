from collections import OrderedDict
import copy
import os
from textwrap import fill, dedent
from dash.development.base_component import _explicitize_args
from dash.exceptions import NonExistentEventException
from ._all_keywords import python_keywords
from ._collect_nodes import collect_nodes, filter_base_nodes
from .base_component import Component
def map_js_to_py_types_flow_types(type_object):
    """Mapping from the Flow js types to the Python type."""
    return dict(array=lambda: 'list', boolean=lambda: 'boolean', number=lambda: 'number', string=lambda: 'string', Object=lambda: 'dict', any=lambda: 'bool | number | str | dict | list', Element=lambda: 'dash component', Node=lambda: 'a list of or a singular dash component, string or number', union=lambda: ' | '.join((js_to_py_type(subType) for subType in type_object['elements'] if js_to_py_type(subType) != '')), Array=lambda: 'list' + (f' of {js_to_py_type(type_object['elements'][0])}s' if js_to_py_type(type_object['elements'][0]) != '' else ''), signature=lambda indent_num: 'dict with keys:\n' + '\n'.join((create_prop_docstring(prop_name=prop['key'], type_object=prop['value'], required=prop['value']['required'], description=prop['value'].get('description', ''), default=prop.get('defaultValue'), indent_num=indent_num + 2, is_flow_type=True) for prop in type_object['signature']['properties'])))