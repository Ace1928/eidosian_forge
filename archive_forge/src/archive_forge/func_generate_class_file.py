from collections import OrderedDict
import copy
import os
from textwrap import fill, dedent
from dash.development.base_component import _explicitize_args
from dash.exceptions import NonExistentEventException
from ._all_keywords import python_keywords
from ._collect_nodes import collect_nodes, filter_base_nodes
from .base_component import Component
def generate_class_file(typename, props, description, namespace, prop_reorder_exceptions=None, max_props=None):
    """Generate a Python class file (.py) given a class string.
    Parameters
    ----------
    typename
    props
    description
    namespace
    prop_reorder_exceptions
    Returns
    -------
    """
    import_string = '# AUTO GENERATED FILE - DO NOT EDIT\n\n' + 'from dash.development.base_component import ' + 'Component, _explicitize_args\n\n\n'
    class_string = generate_class_string(typename, props, description, namespace, prop_reorder_exceptions, max_props)
    file_name = f'{typename:s}.py'
    file_path = os.path.join(namespace, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(import_string)
        f.write(class_string)
    print(f'Generated {file_name}')