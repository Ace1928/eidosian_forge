import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def get_r_type(type_object, is_flow_type=False, indent_num=0):
    """
    Convert JS types to R types for the component definition
    Parameters
    ----------
    type_object: dict
        react-docgen-generated prop type dictionary
    is_flow_type: bool
    indent_num: int
        Number of indents to use for the docstring for the prop
    Returns
    -------
    str
        Python type string
    """
    js_type_name = type_object['name']
    js_to_r_types = get_r_prop_types(type_object=type_object)
    if 'computed' in type_object and type_object['computed'] or type_object.get('type', '') == 'function':
        return ''
    if js_type_name in js_to_r_types:
        prop_type = js_to_r_types[js_type_name]()
        return prop_type
    return ''